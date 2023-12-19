import torch
def check_cuda_memory(epoch, num_epochs, types):
    torch.cuda.empty_cache()  # Clear the cache to obtain accurate memory usage information.
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # Allocated memory, converted to GB
    cached_memory = torch.cuda.memory_reserved() / 1024**3  # cached memory, converted to GB
    log = ""
    log += f"{types} cuda memory:"
    log += f"Epoch [{epoch}/{num_epochs}]:"
    log += f"alloc: {allocated_memory:.4f} GB"
    log += f"cached: {cached_memory:.4f} GB"
    print(log)

def check_cuda_forward(info="None"):
    torch.cuda.empty_cache()  # Clear the cache to obtain accurate memory usage information.
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # Allocated memory, converted to GB
    cached_memory = torch.cuda.memory_reserved() / 1024**3  # cached memory, converted to GB
    log = ""
    log += f"alloc: {allocated_memory:.4f} GB"
    log += f"cached: {cached_memory:.4f} GB"
    print(info)
    print(log)