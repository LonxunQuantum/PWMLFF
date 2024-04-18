import torch
import gc

def check_cuda_memory(epoch, num_epochs, types):
    torch.cuda.empty_cache()  # Clear the cache to obtain accurate memory usage information.
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # Allocated memory, converted to GB
    cached_memory = torch.cuda.memory_cached() / 1024**3  # cached memory, converted to GB
    log = ""
    log += f"{types} cuda memory:"
    log += f"Epoch [{epoch}/{num_epochs}]:"
    log += f"alloc: {allocated_memory:.4f} GB"
    log += f"cached: {cached_memory:.4f} GB"
    print(log)

def find_tensor_memory():
    # 获取所有对象的列表
    objects = gc.get_objects()
    # 过滤出PyTorch张量
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    # 打印张量的大小和内存占用
    for t in tensors:
        print("Size:", t.size(), "Memory:", t.element_size() * t.nelement())