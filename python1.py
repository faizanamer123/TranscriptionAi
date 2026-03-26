import os
import shutil
import asyncio
import logging
from faster_whisper import WhisperModel
import torch
import psutil
import uvicorn
import subprocess
import multiprocessing
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TranscribeAI")


def persistent_worker(task_queue, shared_state, job_to_worker):
    logging.info(f"Worker process {os.getpid()} starting...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel("base", device=device, compute_type="float32")
        pid = os.getpid()

        while True:
            try:
                job_id, path = task_queue.get()
                if job_id is None:
                    task_queue.task_done()
                    break

                job_to_worker[job_id] = pid
                segments, info = model.transcribe(path, beam_size=5)
                total_duration = info.duration

                state = shared_state.get(job_id, {})
                state.update({"duration": total_duration, "status": "processing", "progress": 10})
                shared_state[job_id] = state

                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
                    if total_duration > 0:
                        real_progress = min(99, int((segment.end / total_duration) * 100))
                        state = shared_state.get(job_id, {})
                        if real_progress > state.get("progress", 0):
                            state.update({"progress": real_progress})
                            shared_state[job_id] = state

                full_text = " ".join(text_parts).strip()
                state = shared_state.get(job_id, {})
                state.update({"status": "completed", "progress": 100, "result": full_text})
                shared_state[job_id] = state

            except Exception as e:
                logging.error(f"Error in worker {os.getpid()} for job {job_id}: {str(e)}")
                if job_id:
                    state = shared_state.get(job_id, {})
                    state.update({"status": "failed", "error": str(e)})
                    shared_state[job_id] = state
            finally:
                if job_id:
                    if job_id in job_to_worker:
                        del job_to_worker[job_id]
                    task_queue.task_done()
    except Exception as e:
        logging.critical(f"Worker {os.getpid()} crashed: {str(e)}")


def main():
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    task_queue = multiprocessing.JoinableQueue()
    worker_pool = []
    job_to_worker = manager.dict()

    CPU_CORES = os.cpu_count() or 4
    GPU_AVAILABLE = torch.cuda.is_available()

    # FIX 1: More generous BASE_WORKERS calculation
    # CPU: 1 worker per 2 cores, min 2, max 4
    # GPU: allow up to 6 since GPU handles the heavy lifting
    if GPU_AVAILABLE:
        BASE_WORKERS = min(6, max(2, CPU_CORES // 2))
    else:
        BASE_WORKERS = min(4, max(2, CPU_CORES // 2))

    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv"}
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))
    MAX_JOB_HISTORY = 50

    # FIX 2: Use a mutable dict so all coroutines share live state, not stale closures
    state = {
        "allowed_workers": BASE_WORKERS,
        "is_paused": False,
        "job_id_counter": 0,
    }
    jobs = []
    active_job_ids = set()
    cpu_history = []

    logger.info("--- Hardware Detection ---")
    logger.info(f"Cores: {CPU_CORES}, GPU: {GPU_AVAILABLE}, Base Workers: {BASE_WORKERS}")

    def sync_worker_pool(target_count):
        nonlocal worker_pool
        worker_pool = [p for p in worker_pool if p.is_alive()]

        current_count = len(worker_pool)
        if current_count < target_count:
            for _ in range(target_count - current_count):
                p = multiprocessing.Process(
                    target=persistent_worker,
                    args=(task_queue, shared_state, job_to_worker)
                )
                p.start()
                worker_pool.append(p)
                logger.info(f"Started new worker process (PID: {p.pid})")
        elif current_count > target_count:
            for _ in range(current_count - target_count):
                task_queue.put((None, None))
                logger.info("Signaled worker process to stop")

    async def dispatcher():
        while True:
            if state["is_paused"]:
                await asyncio.sleep(0.5)
                continue

            # Sync job statuses from shared_state
            for job in jobs:
                if job["id"] in shared_state:
                    s = shared_state[job["id"]]

                    if job["status"] == "processing" and s["status"] in ["completed", "failed", "cancelled"]:
                        active_job_ids.discard(job["id"])
                        logger.info(f"Job {job['id']} finished: {s['status']}. Releasing slot.")

                    job.update(s)

                    if s["status"] in ["completed", "failed", "cancelled"]:
                        if os.path.exists(job["path"]):
                            try:
                                os.remove(job["path"])
                                logger.info(f"Cleaned up: {job['path']}")
                            except Exception as e:
                                logger.warning(f"Failed to delete {job['path']}: {e}")

            # Recalculate active set defensively
            active_job_ids.clear()
            for job in jobs:
                if job["status"] == "processing":
                    active_job_ids.add(job["id"])

            # FIX 3: Read allowed_workers from shared mutable dict — always current
            allowed = state["allowed_workers"]

            # Dispatch waiting jobs up to current capacity
            for job in jobs:
                if len(active_job_ids) >= allowed:
                    break
                if job["status"] == "waiting":
                    job["status"] = "processing"
                    active_job_ids.add(job["id"])
                    shared_state[job["id"]] = {"status": "processing", "progress": 10, "result": "", "error": ""}
                    task_queue.put((job["id"], job["path"]))
                    logger.info(
                        f"Dispatched job {job['id']} ({job['file']}). "
                        f"Active: {len(active_job_ids)}/{allowed}"
                    )

            await asyncio.sleep(0.5)  # FIX 4: Faster dispatch loop (was 1s)

    async def cpu_monitor():
        while True:
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent

            cpu_history.append(cpu)
            if len(cpu_history) > 5:  # FIX 5: Larger window = less jitter
                cpu_history.pop(0)

            avg_cpu = sum(cpu_history) / len(cpu_history)
            current = state["allowed_workers"]
            new_limit = current

            # FIX 6: Raised thresholds — don't panic-scale on normal transcription load
            if cpu > 95 or memory > 95:
                # True emergency only
                new_limit = max(1, current - 1)
            elif avg_cpu > 85 or memory > 88:
                new_limit = max(1, current - 1)
            elif avg_cpu < 60 and memory < 80:
                # Healthy headroom — scale up toward BASE_WORKERS
                new_limit = min(current + 1, BASE_WORKERS)
            # else: stay stable

            if new_limit != current:
                logger.info(
                    f"Scaling workers {current} → {new_limit} "
                    f"(CPU: {cpu:.1f}%, avg: {avg_cpu:.1f}%, MEM: {memory:.1f}%)"
                )
                state["allowed_workers"] = new_limit
                sync_worker_pool(new_limit)

            await asyncio.sleep(3)  # FIX 7: Slower monitor = less thrashing

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("System initializing...")
        # FIX 8: Spawn full BASE_WORKERS pool immediately, don't wait for monitor
        sync_worker_pool(BASE_WORKERS)
        logger.info(f"Worker pool started with {BASE_WORKERS} workers.")
        asyncio.create_task(dispatcher())
        asyncio.create_task(cpu_monitor())
        yield
        logger.info("System shutting down...")
        for p in worker_pool:
            if p.is_alive():
                logger.info(f"Terminating worker {p.pid}")
                p.terminate()

    app = FastAPI(
        title="TranscribeAI Enterprise API",
        description="High-performance parallel transcription system",
        version="1.0.0",
        lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/upload")
    async def upload(files: list[UploadFile]):
        os.makedirs("uploads", exist_ok=True)

        newly_added = []
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                logger.warning(f"Rejected invalid file type: {file.filename}")
                continue

            if file.size and file.size > MAX_FILE_SIZE:
                logger.warning(f"Rejected file too large: {file.filename} ({file.size} bytes)")
                continue

            safe_filename = os.path.basename(file.filename)
            job_id = state["job_id_counter"]
            path = os.path.join("uploads", f"{job_id}_{safe_filename}")

            try:
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            except Exception as e:
                logger.error(f"Error saving {file.filename}: {e}")
                continue

            job = {
                "id": job_id,
                "file": safe_filename,
                "path": path,
                "status": "waiting",
                "progress": 0,
                "result": "",
                "duration": 0.0,
                "error": ""
            }

            jobs.append(job)
            newly_added.append(job)
            state["job_id_counter"] += 1

        if not newly_added:
            raise HTTPException(status_code=400, detail="No valid audio/video files provided or files too large.")

        logger.info(f"Batch upload: {len(newly_added)} jobs queued.")
        return {"status": "uploaded", "jobs": newly_added}

    @app.get("/status")
    async def get_status():
        return {
            "cpu_usage": psutil.cpu_percent(),
            "max_workers": state["allowed_workers"],
            "base_workers": BASE_WORKERS,
            "active_workers": len(active_job_ids),
            "queue_size": len([j for j in jobs if j["status"] == "waiting"]),
            "is_paused": state["is_paused"],
            "jobs": jobs[::-1][:MAX_JOB_HISTORY]
        }

    @app.post("/pause")
    async def pause():
        state["is_paused"] = True
        for p in worker_pool:
            try:
                psutil.Process(p.pid).suspend()
            except Exception as e:
                logger.error(f"Failed to suspend worker {p.pid}: {e}")
        logger.info("System paused.")
        return {"status": "paused"}

    @app.post("/resume")
    async def resume():
        state["is_paused"] = False
        for p in worker_pool:
            try:
                psutil.Process(p.pid).resume()
            except Exception as e:
                logger.error(f"Failed to resume worker {p.pid}: {e}")
        logger.info("System resumed.")
        return {"status": "resumed"}

    @app.post("/cancel/{job_id}")
    async def cancel_job(job_id: int):
        nonlocal worker_pool
        job_to_cancel = next((j for j in jobs if j["id"] == job_id), None)

        if not job_to_cancel:
            raise HTTPException(status_code=404, detail="Job not found")

        if job_to_cancel["status"] == "waiting":
            job_to_cancel["status"] = "cancelled"
            logger.info(f"Job {job_id} cancelled (was waiting).")
            return {"status": "cancelled", "job": job_to_cancel}

        elif job_to_cancel["status"] == "processing":
            job_to_cancel["status"] = "cancelled"
            if job_id in job_to_worker:
                target_pid = job_to_worker.pop(job_id)
                for i, p in enumerate(worker_pool):
                    if p.pid == target_pid:
                        p.kill()
                        logger.info(f"Killed worker {p.pid} for job {job_id}.")
                        worker_pool.pop(i)
                        sync_worker_pool(state["allowed_workers"])
                        break

            active_job_ids.discard(job_id)
            if os.path.exists(job_to_cancel["path"]):
                try:
                    os.remove(job_to_cancel["path"])
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

            logger.info(f"Job {job_id} cancelled (was processing).")
            return {"status": "cancelled", "job": job_to_cancel}

        return {"status": "already_completed", "job": job_to_cancel}

    @app.delete("/clear")
    async def clear():
        nonlocal worker_pool
        logger.info("Clearing all jobs and resetting workers...")
        for p in worker_pool:
            try:
                p.kill()
            except:
                pass
        worker_pool = []
        sync_worker_pool(state["allowed_workers"])

        jobs.clear()
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
            except:
                break

        active_job_ids.clear()
        return {"status": "cleared"}

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()