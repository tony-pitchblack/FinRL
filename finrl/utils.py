import platform
import subprocess
import os
from functools import wraps

def get_cpu_info():
    system_name = platform.system()

    if system_name == "Windows":
        cpu_name = subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split("\n")[1]
        cpu_count = os.cpu_count()
    elif system_name == "Linux":
        cpu_name = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip().split(':')[1].strip()
        cpu_count = os.cpu_count()
    elif system_name == "Darwin":  # macOS is identified as "Darwin"
        cpu_name = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
        cpu_count = os.cpu_count()
    else:
        raise NotImplementedError(f"Platform {system_name} is not supported.")

    return cpu_name, cpu_count, system_name

import GPUtil

def get_gpu_info():
    gpus = GPUtil.getGPUs()

    gpu_count = len(gpus)
    gpu_name = gpus[0].name if gpu_count > 0 else "-"

    return gpu_name, gpu_count

import pandas as pd
from time import perf_counter

def benchmark_exec_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model_name = kwargs['model_name']
        if kwargs['drl_lib'] == 'rllib':
            total_timesteps = kwargs['total_episodes'] * kwargs['rllib_params']['train_batch_size']
        elif kwargs['drl_lib'] == 'elegantrl':
            total_timesteps = kwargs['break_step']
        elif kwargs['drl_lib'] == 'stable_baselines3':
            total_timesteps = kwargs['total_timesteps']
        else:
            raise NotImplementedError

        start = perf_counter()
        *output, info = func(*args, **kwargs)
        end = perf_counter()

        exec_time = pd.Timedelta(seconds=end - start)

        cpu_name, cpu_count, system_name = get_cpu_info()
        gpu_name, gpu_count = get_gpu_info()

        data = {
            "func_name": func.__name__,
            "exec_time": exec_time,
            "data_shape": info['data_shape'][0],
            "num_stocks": info['num_stocks'],
            "model_name": model_name,
            "total_timesteps": total_timesteps,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "cpu_count": cpu_count,
            "cpu_name": cpu_name,
            "system_name": system_name,
        }

        print('\nBenchmark results:')
        series = pd.Series(data, name="benchmark_results")
        print(series)  # TODO: log in file

        return output

    return wrapper