from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch


def build_model(config):
    """
    Build a model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        torch.nn.Module: The constructed model.
    """
    # Extract model type and parameters from the config
    model_type = config.model.type
    model_params = config.model.params

    # Dynamically import the model class based on the type
    module = __import__(f"src.models.{model_type}", fromlist=[model_type])
    model_class = getattr(module, model_type)

    # Create an instance of the model with the specified parameters
    return model_class(**model_params)


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
