import os
import shutil
import asyncio
import logging
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# Load environment variables
load_dotenv()
import torch
import psutil
import uvicorn
import subprocess
import multiprocessing
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TranscribeAI")

# Security Configuration
API_KEY = os.getenv("API_KEY", "dev-key-123")  # Change this in production!
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

async def verify_api_key(request: Request):
    if request.method == "OPTIONS":
        return
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt from {request.client.host}")
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")

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
                state.update({
                    "status": "completed",
                    "progress": 100,
                    "result": full_text
                })
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
    # Global State and Configuration
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    task_queue = multiprocessing.JoinableQueue()
    worker_pool = []
    job_to_worker = manager.dict()

    CPU_CORES = os.cpu_count() or 4
    GPU_AVAILABLE = torch.cuda.is_available()
    BASE_WORKERS = max(1, CPU_CORES // 4)
    if GPU_AVAILABLE:
        BASE_WORKERS = min(BASE_WORKERS, 2)
    else:
        BASE_WORKERS = min(BASE_WORKERS, 2)

    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv"}
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))
    MAX_JOB_HISTORY = 50

    current_allowed_workers = BASE_WORKERS
    job_id_counter = 0
    jobs = []
    active_job_ids = set()
    cpu_history = []
    is_paused = False

    logger.info("--- Hardware Detection ---")
    logger.info(f"Cores: {CPU_CORES}, GPU: {GPU_AVAILABLE}, Base Workers: {BASE_WORKERS}")

    # Worker and System Functions
    def get_audio_duration(path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not determine audio duration for {path}: {e}")
            return 0.0

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
        nonlocal is_paused
        while True: 
            if is_paused:
                await asyncio.sleep(0.5)
                continue

            # Update local jobs list from shared state
            for job in jobs:
                if job["id"] in shared_state:
                    state = shared_state[job["id"]]
                    if job["status"] == "processing" and state["status"] in ["completed", "failed", "cancelled"]:
                        active_job_ids.discard(job["id"])
                    
                    job.update(state)
                    
                    if state["status"] in ["completed", "failed", "cancelled"]:
                        if os.path.exists(job["path"]):
                            try: 
                                os.remove(job["path"])
                                logger.info(f"Cleaned up temporary file: {job['path']}")
                            except Exception as e: 
                                logger.warning(f"Failed to delete {job['path']}: {e}")

            # Assign new jobs
            for job in jobs:
                if job["status"] == "waiting" and len(active_job_ids) < current_allowed_workers:
                    if job.get("status") == "cancelled":
                        continue
                    job["status"] = "processing"
                    active_job_ids.add(job["id"])
                    shared_state[job["id"]] = {"status": "processing", "progress": 10, "result": "", "error": ""}
                    task_queue.put((job["id"], job["path"]))
                    logger.info(f"Assigned job {job['id']} ({job['file']}) to worker pool.")
            
            await asyncio.sleep(1)

    async def cpu_monitor(): 
        nonlocal current_allowed_workers, cpu_history
        while True: 
            cpu = psutil.cpu_percent(interval=None) 
            memory = psutil.virtual_memory().percent
            
            cpu_history.append(cpu)
            if len(cpu_history) > 3:
                cpu_history.pop(0)
            
            avg_cpu = sum(cpu_history) / len(cpu_history)
            
            new_limit = current_allowed_workers
            if cpu > 80 or avg_cpu > 70 or memory > 85:
                new_limit = 1
            elif avg_cpu < 30 and memory < 60 and not GPU_AVAILABLE:
                new_limit = min(CPU_CORES // 2, 4)
            elif avg_cpu < 45 and memory < 75:
                new_limit = BASE_WORKERS
            else:
                new_limit = min(current_allowed_workers, BASE_WORKERS)
                
            if new_limit != current_allowed_workers:
                logger.info(f"Scaling worker pool from {current_allowed_workers} to {new_limit} based on system load (CPU: {cpu}%, MEM: {memory}%)")
                current_allowed_workers = new_limit
                sync_worker_pool(current_allowed_workers)
                
            await asyncio.sleep(2) # Monitor every 2 seconds for stability

    # FastAPI Lifespan Management
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("System initializing...")
        sync_worker_pool(current_allowed_workers)
        asyncio.create_task(dispatcher())
        asyncio.create_task(cpu_monitor())
        yield
        logger.info("System shutting down...")
        for p in worker_pool:
            if p.is_alive():
                logger.info(f"Terminating worker process {p.pid}")
                p.terminate()

    # FastAPI App Initialization
    app = FastAPI(
        title="TranscribeAI Enterprise API",
        description="High-performance parallel transcription system",
        version="1.0.0",
        lifespan=lifespan,
        dependencies=[Depends(verify_api_key)]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Endpoints
    @app.post("/upload")
    async def upload(files: list[UploadFile]):
        nonlocal job_id_counter
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
            path = os.path.join("uploads", f"{job_id_counter}_{safe_filename}")
            
            try:
                with open(path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            except Exception as e:
                logger.error(f"Error saving uploaded file {file.filename}: {e}")
                continue

            job = {
                "id": job_id_counter,
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
            job_id_counter += 1

        if not newly_added:
            raise HTTPException(status_code=400, detail="No valid audio/video files provided or files too large.")

        logger.info(f"Batch upload successful: {len(newly_added)} jobs created.")
        return {"status": "uploaded", "jobs": newly_added}

    @app.get("/status")
    async def get_status():
        return {
            "cpu_usage": psutil.cpu_percent(),
            "max_workers": current_allowed_workers,
            "active_workers": len(active_job_ids),
            "queue_size": len([j for j in jobs if j["status"] == "waiting"]),
            "is_paused": is_paused,
            "jobs": jobs[::-1][:MAX_JOB_HISTORY] # Return only recent history
        }

    @app.post("/pause")
    async def pause():
        nonlocal is_paused
        is_paused = True
        for p in worker_pool:
            try:
                ps_proc = psutil.Process(p.pid)
                ps_proc.suspend()
            except Exception as e: 
                logger.error(f"Failed to suspend worker {p.pid}: {e}")
        logger.info("System execution paused by user.")
        return {"status": "paused"}

    @app.post("/resume")
    async def resume():
        nonlocal is_paused
        is_paused = False
        for p in worker_pool:
            try:
                ps_proc = psutil.Process(p.pid)
                ps_proc.resume()
            except Exception as e: 
                logger.error(f"Failed to resume worker {p.pid}: {e}")
        logger.info("System execution resumed by user.")
        return {"status": "resumed"}

    @app.post("/cancel/{job_id}")
    async def cancel_job(job_id: int):
        nonlocal worker_pool
        job_to_cancel = next((job for job in jobs if job["id"] == job_id), None)

        if not job_to_cancel:
            raise HTTPException(status_code=404, detail="Job not found")

        if job_to_cancel["status"] == "waiting":
            job_to_cancel["status"] = "cancelled"
            logger.info(f"Job {job_id} cancelled while waiting.")
            return {"status": "cancelled", "job": job_to_cancel}
        
        elif job_to_cancel["status"] == "processing":
            job_to_cancel["status"] = "cancelled"
            if job_id in job_to_worker:
                target_pid = job_to_worker.pop(job_id)
                for i, p in enumerate(worker_pool):
                    if p.pid == target_pid:
                        p.kill()
                        logger.info(f"Worker {p.pid} killed to cancel job {job_id}.")
                        worker_pool.pop(i)
                        sync_worker_pool(current_allowed_workers)
                        break
            
            active_job_ids.discard(job_id)
            if os.path.exists(job_to_cancel["path"]):
                try: 
                    os.remove(job_to_cancel["path"])
                    logger.info(f"Cleaned up temporary file for cancelled job: {job_to_cancel['path']}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file for cancelled job: {e}")
            
            logger.info(f"Active job {job_id} cancelled and worker reset.")
            return {"status": "cancelled", "job": job_to_cancel}
            
        return {"status": "already_completed", "job": job_to_cancel}

    @app.delete("/clear")
    async def clear():
        nonlocal jobs, worker_pool
        logger.info("Clearing all job history and resetting worker pool...")
        for p in worker_pool:
            try: p.kill()
            except: pass
        worker_pool = []
        sync_worker_pool(current_allowed_workers)

        jobs = []
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
            except: break
            
        active_job_ids.clear()
        return {"status": "cleared"}

    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# Main Execution Block
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
