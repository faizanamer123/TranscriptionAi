import os, shutil, asyncio, logging, multiprocessing, psutil, torch, uvicorn, time
from faster_whisper import WhisperModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TranscribeAI")

def persistent_worker(task_queue, shared_state, job_to_worker, idx):
    try:
        dev = f"cuda:{idx % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        model = WhisperModel("base", device=dev, compute_type="float16" if "cuda" in dev else "float32")
        while True:
            try:
                job_id, path = task_queue.get()
                if job_id is None: break
                if shared_state.get(f"cancel_{job_id}"):
                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                    task_queue.task_done(); continue
                job_to_worker[job_id] = os.getpid()
                segments, info = model.transcribe(path, beam_size=5)
                txt = []
                for s in segments:
                    if shared_state.get(f"cancel_{job_id}"): break
                    # Pause handling: stop worker and signal paused state
                    if shared_state.get("is_paused", False):
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "paused"}
                        while shared_state.get("is_paused", False):
                            time.sleep(0.5)
                            if shared_state.get(f"cancel_{job_id}"): break
                        if not shared_state.get(f"cancel_{job_id}"):
                            shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "processing"}
                    if shared_state.get(f"cancel_{job_id}"): break

                    txt.append(s.text)
                    prog = min(99, int((s.end / info.duration) * 100)) if info.duration > 0 else 10
                    if not shared_state.get(f"cancel_{job_id}"):
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "progress": prog, "status": "processing"}
                if shared_state.get(f"cancel_{job_id}"):
                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                else:
                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "completed", "progress": 100, "result": " ".join(txt).strip()}
                del job_to_worker[job_id]; task_queue.task_done()
            except Exception as e:
                logger.error(f"Worker {os.getpid()} task error: {e}")
    except Exception as e: logger.error(f"Worker {os.getpid()} initialization error: {e}")

def detect_hardware():
    l_cores, p_cores = os.cpu_count() or 4, psutil.cpu_count(logical=False) or 4
    gpu_on, gpu_count = torch.cuda.is_available(), torch.cuda.device_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    mem_lim = max(1, int((ram_gb - 2.5) // 1.2))
    batt = psutil.sensors_battery()
    on_batt = batt.power_plugged is False if batt else False
    base = min(mem_lim, p_cores, (gpu_count*3 + p_cores//4) if gpu_on else max(2, int(p_cores//1.2)))
    return l_cores, p_cores, gpu_on, gpu_count, base, ram_gb, on_batt, mem_lim

L_CORES, P_CORES, GPU_ON, GPU_COUNT, BASE_WORKERS, RAM_GB, ON_BATT, MEM_LIM = detect_hardware()
manager, shared_state, task_queue, job_to_worker, worker_pool, lock = None, None, None, None, [], None
jobs, active_ids, cpu_hist, last_scale = [], set(), [], 0

def sync_pool(target):
    global worker_pool
    worker_pool = [p for p in worker_pool if p.is_alive()]
    curr = len(worker_pool)
    if curr < target:
        for i in range(curr, target):
            p = multiprocessing.Process(target=persistent_worker, args=(task_queue, shared_state, job_to_worker, i), daemon=True)
            p.start(); worker_pool.append(p)
    elif curr > target:
        for _ in range(curr - target): task_queue.put((None, None))

async def dispatcher():
    while True:
        try:
            # Always update job statuses and active_ids even when paused
            for j in jobs:
                if j["id"] in shared_state:
                    s = shared_state[j["id"]]
                    if j["status"] in ["cancelled", "completed", "failed"] and s.get("status") in ["processing", "paused"]:
                        continue
                    j.update(s)
                    if j["status"] in ["completed", "failed", "cancelled"] and os.path.exists(j["path"]):
                        try: os.remove(j["path"])
                        except: pass
            
            active_ids.clear(); active_ids.update([j["id"] for j in jobs if j["status"] in ["processing", "paused"]])
            
            # Only dispatch new jobs if NOT paused
            if not shared_state.get("is_paused", False):
                allowed = shared_state.get("allowed_workers", BASE_WORKERS)
                for j in [j for j in jobs if j["status"] == "waiting"]:
                    if len(active_ids) >= allowed: break
                    j["status"] = "processing"; active_ids.add(j["id"])
                    shared_state[j["id"]] = {"status": "processing", "progress": 10}
                    task_queue.put((j["id"], j["path"]))
        except (EOFError, ConnectionResetError, BrokenPipeError): break
        except Exception as e: logger.error(f"Dispatcher error: {e}")
        await asyncio.sleep(0.5)

async def cpu_monitor():
    global last_scale
    while True:
        try:
            batt = psutil.sensors_battery()
            on_b = batt.power_plugged is False if batt else False
            shared_state["is_on_battery"] = on_b
            cpu, mem, q_size = psutil.cpu_percent(), psutil.virtual_memory().percent, len([j for j in jobs if j["status"] == "waiting"])
            cpu_hist.append(cpu)
            if len(cpu_hist) > 10: cpu_hist.pop(0)
            avg_cpu, curr, now = sum(cpu_hist)/len(cpu_hist), shared_state["allowed_workers"], asyncio.get_event_loop().time()
            lim = curr
            if cpu > 95 or mem > 92: lim, last_scale = max(1, curr-1), now
            elif (now - last_scale) > 10:
                if on_b:
                    b_cap = max(1, BASE_WORKERS // 2)
                    lim = b_cap if curr > b_cap else (min(curr+1, b_cap) if avg_cpu < 20 else curr)
                else:
                    if curr < BASE_WORKERS and avg_cpu < 65: lim = curr + 1
                    elif (q_size > 0 or avg_cpu < 30) and mem < 75: lim = min(curr+1, P_CORES, MEM_LIM)
                    elif avg_cpu > 80: lim = max(BASE_WORKERS, curr-1)
            if lim != curr:
                shared_state["allowed_workers"] = lim; sync_pool(lim); last_scale = now
        except (EOFError, ConnectionResetError, BrokenPipeError): break
        except Exception as e: logger.error(f"CPU Monitor error: {e}")
        await asyncio.sleep(4)

@asynccontextmanager
async def lifespan(app):
    global manager, shared_state, task_queue, job_to_worker, lock
    manager = multiprocessing.Manager(); shared_state, task_queue, job_to_worker, lock = manager.dict(), manager.JoinableQueue(), manager.dict(), manager.Lock()
    shared_state.update({"allowed_workers": BASE_WORKERS, "is_paused": False, "job_id_counter": 0, "is_on_battery": ON_BATT})
    sync_pool(BASE_WORKERS); asyncio.create_task(dispatcher()); asyncio.create_task(cpu_monitor())
    yield
    for p in worker_pool:
        if p.is_alive(): p.terminate()
    manager.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def ui(): return FileResponse("index.html")

@app.post("/upload")
async def upload(files: list[UploadFile]):
    os.makedirs("uploads", exist_ok=True); added = []
    with lock:
        for f in files:
            if not f.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4', '.mov', '.mkv')): continue
            jid = shared_state["job_id_counter"]
            # Ensure the counter exists and is updated atomically
            shared_state["job_id_counter"] = jid + 1
            path = os.path.join("uploads", f"{jid}_{f.filename}")
            with open(path, "wb") as buf: shutil.copyfileobj(f.file, buf)
            j = {"id": jid, "file": f.filename, "path": path, "status": "waiting", "progress": 0, "result": "", "error": ""}
            jobs.append(j); added.append(j)
    if not added: raise HTTPException(400, "No valid files")
    return {"status": "uploaded", "jobs": added}

@app.get("/status")
async def status():
    return {"cpu_usage": psutil.cpu_percent(), "max_workers": shared_state["allowed_workers"], "base_workers": BASE_WORKERS, "active_workers": len(active_ids), "queue_size": len([j for j in jobs if j["status"] == "waiting"]), "is_paused": shared_state["is_paused"], "is_on_battery": shared_state["is_on_battery"], "total_ram": f"{RAM_GB:.1f} GB", "jobs": jobs[::-1][:50]}

@app.post("/pause")
async def pause():
    shared_state["is_paused"] = True
    for j in jobs:
        if j["status"] == "processing":
            j["status"] = "paused"
            shared_state[j["id"]] = {**shared_state.get(j["id"], {}), "status": "paused"}
    return {"status": "paused"}

@app.post("/resume")
async def resume():
    shared_state["is_paused"] = False
    for j in jobs:
        if j["status"] == "paused":
            j["status"] = "processing"
            shared_state[j["id"]] = {**shared_state.get(j["id"], {}), "status": "processing"}
    return {"status": "resumed"}

@app.post("/cancel/{jid}")
async def cancel(jid: int):
    j = next((j for j in jobs if j["id"] == jid), None)
    if not j: raise HTTPException(404)
    if j["status"] in ["waiting", "processing"]:
        shared_state[f"cancel_{jid}"] = True
        j["status"] = "cancelled"; j["progress"] = 0
        shared_state[jid] = {**shared_state.get(jid, {}), "status": "cancelled", "progress": 0}
    return {"status": "cancelled"}

@app.delete("/clear")
async def clear():
    for p in worker_pool:
        try: p.kill()
        except: pass
    sync_pool(shared_state["allowed_workers"]); jobs.clear(); active_ids.clear(); return {"status": "cleared"}

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run("python1:app", host="0.0.0.0", port=8002, reload=True)
