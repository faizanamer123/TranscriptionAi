import os
import psutil
import torch

def detect_hardware():
    logical_cores = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or 4
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    total_ram_gb = psutil.virtual_memory().total / (1024**3)

    if gpu_available:
        base_workers = min(physical_cores * 2, gpu_count * 3)
    else:
        base_workers = max(2, physical_cores // 2)

    memory_limit_gb = max(1, int((total_ram_gb - 2) // 1.2))
    battery = psutil.sensors_battery()
    on_battery = battery.power_plugged is False if battery else False

    return (
        logical_cores,
        physical_cores,
        gpu_available,
        gpu_count,
        base_workers,
        total_ram_gb,
        memory_limit_gb,
        on_battery,
    )
