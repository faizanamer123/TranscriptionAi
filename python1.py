
import os
import shutil
import asyncio
import logging
import subprocess
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import torch
import psutil
import uvicorn
import multiprocessing
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# --- Global Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TranscribeAI")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# --- Worker Process Definition ---
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
                state = shared_state.get(job_id, {})
                state.update({"status": "processing", "progress": 5})
                shared_state[job_id] = state

                segments, info = model.transcribe(path, beam_size=5)
                
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
                    if info.duration > 0:
                        progress = min(99, int((segment.end / info.duration) * 100))
                        state = shared_state.get(job_id, {})
                        if progress > state.get("progress", 0):
                            state.update({"progress": progress})
                            shared_state[job_id] = state

                full_text = " ".join(text_parts).strip()
                state = shared_state.get(job_id, {})
                state.update({"status": "completed", "progress": 100, "result": full_text, "duration": info.duration})
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

# --- Main Application Logic ---
def main():
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    task_queue = multiprocessing.JoinableQueue()
    worker_pool = []
    job_to_worker = manager.dict()

    CPU_CORES = os.cpu_count() or 4
    GPU_AVAILABLE = torch.cuda.is_available()
    BASE_WORKERS = max(1, CPU_CORES // 4) if not GPU_AVAILABLE else 1

    current_allowed_workers = BASE_WORKERS
    job_id_counter = 0
    jobs = []
    active_job_ids = set()
    is_paused = False

    logger.info(f"--- Hardware Detection: {CPU_CORES} Cores, GPU: {GPU_AVAILABLE}, Base Workers: {BASE_WORKERS} ---")

    def get_audio_duration(path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get duration for {path}: {e}")
            return 0.0

    def sync_worker_pool(target_count):
        nonlocal worker_pool
        worker_pool = [p for p in worker_pool if p.is_alive()]
        current_count = len(worker_pool)

        if current_count < target_count:
            for _ in range(target_count - current_count):
                p = multiprocessing.Process(target=persistent_worker, args=(task_queue, shared_state, job_to_worker))
                p.start()
                worker_pool.append(p)
                logger.info(f"Started new worker (PID: {p.pid})")
        elif current_count > target_count:
            for _ in range(current_count - target_count):
                task_queue.put((None, None))
                logger.info("Signaled worker to stop")

    async def dispatcher():
        nonlocal is_paused
        while True:
            await asyncio.sleep(1)
            if is_paused: continue

            for job in jobs:
                if job["id"] in shared_state:
                    state = shared_state[job["id"]]
                    if job["status"] == "processing" and state["status"] in ["completed", "failed", "cancelled"]:
                        active_job_ids.discard(job["id"])
                    job.update(state)

            for job in jobs:
                if job["status"] == "waiting" and len(active_job_ids) < current_allowed_workers:
                    job["status"] = "processing"
                    active_job_ids.add(job["id"])
                    shared_state[job["id"]] = {"status": "processing", "progress": 0}
                    task_queue.put((job["id"], job["path"]))
                    logger.info(f"Assigned job {job['id']} to worker pool.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("System initializing...")
        sync_worker_pool(current_allowed_workers)
        asyncio.create_task(dispatcher())
        yield
        logger.info("System shutting down...")
        for p in worker_pool:
            if p.is_alive(): p.terminate()

    app = FastAPI(title="TranscribeAI API", version="1.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    @app.post("/upload")
    async def upload(files: list[UploadFile]):
        nonlocal job_id_counter
        os.makedirs("uploads", exist_ok=True)
        newly_added = []
        for file in files:
            path = os.path.join("uploads", f"{job_id_counter}_{os.path.basename(file.filename)}")
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            duration = get_audio_duration(path)

            job = {"id": job_id_counter, "file": file.filename, "path": path, "status": "waiting", "progress": 0, "duration": duration}
            jobs.append(job)
            newly_added.append(job)
            job_id_counter += 1
        return {"status": "uploaded", "jobs": newly_added}

    @app.get("/status")
    async def get_status():
        return {"active_workers": len(active_job_ids), "jobs": jobs[::-1]}

    @app.post("/cancel/{job_id}")
    async def cancel_job(job_id: int):
        job_to_cancel = next((j for j in jobs if j["id"] == job_id), None)
        if not job_to_cancel: raise HTTPException(status_code=404, detail="Job not found")

        if job_to_cancel["status"] == "waiting":
            job_to_cancel["status"] = "cancelled"
            return {"status": "cancelled"}
        
        if job_to_cancel["status"] == "processing":
            job_to_cancel["status"] = "cancelled"
            if job_id in job_to_worker:
                pid = job_to_worker.pop(job_id)
                for p in worker_pool:
                    if p.pid == pid:
                        p.kill()
                        sync_worker_pool(current_allowed_workers)
                        break
            active_job_ids.discard(job_id)
            return {"status": "cancelled"}

        return {"status": "already_completed"}

    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
