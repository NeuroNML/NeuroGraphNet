import torch

# print CUDA memory usage
def print_cuda_memory_usage():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    else:
        print("CUDA is not available.")
def clean_cuda_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    else:
        print("CUDA is not available.")