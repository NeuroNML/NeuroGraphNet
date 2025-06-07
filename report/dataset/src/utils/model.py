import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

def pooling(x, batch, pooling_type):
    if pooling_type == "mean":
        return global_mean_pool(x, batch)
    elif pooling_type == "max":
        return global_max_pool(x, batch)
    elif pooling_type == "sum":
        return global_add_pool(x, batch)
    elif pooling_type == "mean+max":
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
    elif pooling_type == "sum+mean":
        return torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch)], dim=1)
    elif pooling_type == "sum+max":
        return torch.cat([global_add_pool(x, batch), global_max_pool(x, batch)], dim=1)
    elif pooling_type == "sum+mean+max":
        return torch.cat(
            [
                global_add_pool(x, batch),
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")