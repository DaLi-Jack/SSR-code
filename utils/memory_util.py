import os
import psutil

def get_gpu_mem_info(gpu_id=0):
    """
    show gpu memory  (MB)
    :param gpu_id: gpu index
    :return: total memory, used memory, free memory
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(f'gpu_id {gpu_id} not exist!')
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    show all virtual memory in this computer  (MB)
    :return: mem_total, mem_free, mem_process_used
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used


if __name__ == "__main__":
    gpu_id = 0
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id)
    print(f'gpu {gpu_id}: total {gpu_mem_total} MB, used {gpu_mem_used} MB, free {gpu_mem_free} MB')

    cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
    print(f'computer : total {cpu_mem_total} MB, used {cpu_mem_process_used} MB, free {cpu_mem_free} MB')
