import importlib

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
    model_name = config.model.name
    model_params = config.model.params

    # Import model class
    module = importlib.import_module(
        f"src.models.{model_name}"
    )  # loads file src/models/{model_type}.py
    model_class = getattr(
        module, model_name
    )  # Extract the class (model_type) from the module

    return model_class(
        **model_params
    )  # Create an instance of the model with the specified parameters


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
