from __future__ import annotations
from typing import Any
import asyncio
import multiprocessing
import os
import shutil
import time
from contextlib import asynccontextmanager
import psutil
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from hardware import detect_hardware
from settings import JOBS_MAX, logger
from worker import clear_cancel_flag, persistent_worker

(
    _logical_cores,
    P_CORES,
    _gpu_on,
    _gpu_count,
    BASE_WORKERS,
    RAM_GB,
    MEM_LIM,
    ON_BATT,
) = detect_hardware()

manager = None
shared_state = None
task_queue = None
job_to_worker = None
worker_pool = []
lock = None
jobs = []
active_ids = set()
cpu_hist = []
last_scale = 0.0


def prune_jobs():
    # Drop oldest terminal jobs first so the in-memory list stays bounded
    global jobs
    if len(jobs) <= JOBS_MAX:
        return
    terminal = frozenset({"completed", "failed", "cancelled"})
    need = len(jobs) - JOBS_MAX
    terminal_sorted = sorted(
        (j for j in jobs if j["status"] in terminal),
        key=lambda x: x["id"],
    )
    remove_ids = {j["id"] for j in terminal_sorted[:need]}
    if len(remove_ids) < need:
        logger.warning(
            "job cap: need %s removals but only %s terminal jobs",
            need,
            len(terminal_sorted),
        )
    jobs = [j for j in jobs if j["id"] not in remove_ids]


def sync_pool(target: int):
    global worker_pool
    worker_pool = [p for p in worker_pool if p.is_alive()]
    curr = len(worker_pool)
    if curr < target:
        for i in range(curr, target):
            p = multiprocessing.Process(
                target=persistent_worker,
                args=(task_queue, shared_state, job_to_worker, i),
                daemon=True,
            )
            p.start()
            worker_pool.append(p)
    elif curr > target:
        for _ in range(curr - target):
            task_queue.put((None, None))


async def dispatcher():
    while True:
        try:
            async with lock:
                for j in jobs:
                    if j["id"] not in shared_state:
                        continue
                    s = shared_state[j["id"]]
                    if j["status"] in ["cancelled", "completed", "failed"] and s.get(
                        "status"
                    ) in ["processing", "paused"]:
                        continue
                    j.update(s)
                    if j["status"] in ["completed", "failed", "cancelled"] and os.path.exists(
                        j["path"]
                    ):
                        try:
                            os.remove(j["path"])
                        except OSError:
                            pass

                allowed = shared_state.get("allowed_workers", BASE_WORKERS)
                active_jobs = [
                    j
                    for j in jobs
                    if j["status"]
                    not in ["completed", "failed", "cancelled", "waiting"]
                ]
                active_jobs.sort(key=lambda x: x["id"])

                for i, j in enumerate(active_jobs):
                    if i < allowed:
                        if shared_state.get(f"pause_{j['id']}") and not shared_state.get(
                            "is_paused"
                        ):
                            shared_state[f"pause_{j['id']}"] = False
                    else:
                        shared_state[f"pause_{j['id']}"] = True

                active_ids.clear()
                active_ids.update(
                    j["id"]
                    for j in active_jobs
                    if not shared_state.get(f"pause_{j['id']}")
                )

                if not shared_state.get("is_paused", False):
                    for j in [x for x in jobs if x["status"] == "waiting"]:
                        if shared_state.get(f"cancel_{j['id']}"):
                            j["status"] = "cancelled"
                            shared_state[j["id"]] = {
                                **shared_state.get(j["id"], {}),
                                "status": "cancelled",
                                "progress": 0,
                            }
                            clear_cancel_flag(j["id"], shared_state)
                            continue
                        if len(active_ids) >= allowed:
                            break
                        j["status"] = "transcribing"
                        active_ids.add(j["id"])
                        shared_state[j["id"]] = {"status": "transcribing", "progress": 2}
                        task_queue.put((j["id"], j["path"]))
                prune_jobs()
        except (EOFError, ConnectionResetError, BrokenPipeError):
            break
        except Exception as e:
            logger.error(f"Dispatcher: {e}")
        await asyncio.sleep(0.5)


async def cpu_monitor():
    global last_scale
    while True:
        try:
            batt = psutil.sensors_battery()
            on_b = batt.power_plugged is False if batt else False
            shared_state["is_on_battery"] = on_b
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            async with lock:
                q_size = len([j for j in jobs if j["status"] == "waiting"])
            cpu_hist.append(cpu)
            if len(cpu_hist) > 10:
                cpu_hist.pop(0)
            avg_cpu = sum(cpu_hist) / len(cpu_hist) if cpu_hist else 0
            curr = shared_state["allowed_workers"]
            now = time.monotonic()
            lim = curr
            if cpu > 95 or mem > 92:
                lim = max(1, curr - 1)
                last_scale = now
            elif (now - last_scale) > 10:
                if on_b:
                    b_cap = max(1, BASE_WORKERS // 2)
                    lim = b_cap if curr > b_cap else (
                        min(curr + 1, b_cap) if avg_cpu < 20 else curr
                    )
                else:
                    if curr < BASE_WORKERS and avg_cpu < 65:
                        lim = curr + 1
                    elif (q_size > 0 or avg_cpu < 30) and mem < 75:
                        lim = min(curr + 1, P_CORES, MEM_LIM)
                    elif avg_cpu > 80:
                        lim = max(BASE_WORKERS, curr - 1)
            if lim != curr:
                shared_state["allowed_workers"] = lim
                sync_pool(lim)
                last_scale = now
        except (EOFError, ConnectionResetError, BrokenPipeError):
            break
        except Exception as e:
            logger.error(f"CPU monitor: {e}")
        await asyncio.sleep(4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, shared_state, task_queue, job_to_worker, lock, last_scale

    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    task_queue = manager.JoinableQueue()
    job_to_worker = manager.dict()
    lock = asyncio.Lock()
    shared_state.update(
        {
            "allowed_workers": BASE_WORKERS,
            "is_paused": False,
            "job_id_counter": 0,
            "is_on_battery": ON_BATT,
        }
    )
    last_scale = time.monotonic()
    sync_pool(BASE_WORKERS)
    asyncio.create_task(dispatcher())
    asyncio.create_task(cpu_monitor())
    yield
    for p in worker_pool:
        if p.is_alive():
            p.terminate()
    manager.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/download/{filename}")
async def download(filename: str):
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename:
        raise HTTPException(400, "Invalid filename")
    if not safe_filename.endswith(".md"):
        raise HTTPException(400, "Only .md downloads allowed")
    base = os.path.abspath("transcriptions")
    path = os.path.join(base, safe_filename)
    real = os.path.abspath(path)
    try:
        if os.path.commonpath([base, real]) != base:
            raise HTTPException(400, "Invalid path")
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    return FileResponse(
        path, media_type="text/markdown", filename=safe_filename
    )


@app.post("/upload")
async def upload(files: list[UploadFile]):
    os.makedirs("uploads", exist_ok=True)
    added = []
    async with lock:
        for f in files:
            if not f.filename:
                continue
            if not f.filename.lower().endswith(
                (".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv")
            ):
                continue
            jid = shared_state["job_id_counter"]
            shared_state["job_id_counter"] = jid + 1
            path = os.path.join("uploads", f"{jid}_{f.filename}")
            with open(path, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            j = {
                "id": jid,
                "file": f.filename,
                "path": path,
                "status": "waiting",
                "progress": 0,
                "result": "",
                "error": "",
            }
            jobs.append(j)
            added.append(j)
        prune_jobs()
    if not added:
        raise HTTPException(400, "No valid audio/video files")
    return {"status": "uploaded", "jobs": added}


@app.get("/status")
async def status():
    async with lock:
        active_jobs_count = len(
            [
                j
                for j in jobs
                if j["status"]
                not in ["completed", "failed", "cancelled", "waiting", "paused"]
            ]
        )
        queue_size = len([j for j in jobs if j["status"] == "waiting"])
        jobs_snapshot = jobs[::-1][:50]
    return {
        "cpu_usage": psutil.cpu_percent(),
        "max_workers": shared_state["allowed_workers"],
        "base_workers": BASE_WORKERS,
        "active_workers": active_jobs_count,
        "queue_size": queue_size,
        "is_paused": shared_state["is_paused"],
        "is_on_battery": shared_state["is_on_battery"],
        "total_ram": f"{RAM_GB:.1f} GB",
        "jobs": jobs_snapshot,
    }


@app.post("/pause")
async def pause():
    shared_state["is_paused"] = True
    async with lock:
        for j in jobs:
            if j["status"] not in [
                "completed",
                "failed",
                "cancelled",
                "waiting",
                "paused",
            ]:
                j["prev_status"] = j["status"]
                j["status"] = "paused"
                shared_state[j["id"]] = {
                    **shared_state.get(j["id"], {}),
                    "status": "paused",
                }
    return {"status": "paused"}


@app.post("/resume")
async def resume():
    shared_state["is_paused"] = False
    async with lock:
        for j in jobs:
            if j["status"] == "paused":
                restored = j.get("prev_status", "transcribing")
                j["status"] = restored
                shared_state[j["id"]] = {
                    **shared_state.get(j["id"], {}),
                    "status": restored,
                }
    return {"status": "resumed"}


CANCELLABLE = frozenset(
    {
        "waiting",
        "paused",
        "transcribing",
        "aligning",
        "diarizing",
        "finalizing",
    }
)


@app.post("/cancel/{jid}")
async def cancel(jid: int):
    async with lock:
        j = next((x for x in jobs if x["id"] == jid), None)
        if not j:
            raise HTTPException(404, "Job not found")

        prev = j["status"]
        if prev not in CANCELLABLE:
            raise HTTPException(
                400,
                detail=f"Cannot cancel job in state '{prev}'",
            )

        logger.info(f"Cancel {jid} (was {prev})")
        shared_state[f"cancel_{jid}"] = True
        j["status"] = "cancelled"
        j["progress"] = 0
        shared_state[jid] = {
            **shared_state.get(jid, {}),
            "status": "cancelled",
            "progress": 0,
        }
        if jid in active_ids:
            active_ids.discard(jid)
        if prev == "waiting":
            logger.info(f"Job {jid} cancelled from queue")

    return {"status": "cancelled", "job_id": jid}


@app.delete("/clear")
async def clear():
    for p in worker_pool:
        try:
            p.kill()
        except Exception:
            pass
    sync_pool(shared_state["allowed_workers"])
    async with lock:
        jobs.clear()
        active_ids.clear()
    return {"status": "cleared"}


def run():
    multiprocessing.freeze_support()
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
    )


if __name__ == "__main__":
    run()
