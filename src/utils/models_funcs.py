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
    model_name = config.model.name
    model_type = config.model.type
    model_params = config.model.params

    # Encoder + gnn models
    if model_type == "encoder_gnn":
        # Load encoder
        encoder = config.model.encoder
        encoder_module = importlib.import_module(f"src.models.{encoder.type}")
        encoder_class = getattr(encoder_module, encoder.name)
        encoder = encoder_class(**encoder.params)

        # Load gnn
        gnn = config.model.gnn
        gnn_module = importlib.import_module(f"src.models.{gnn.type}")
        gnn_class = getattr(gnn_module, gnn.name)
        gnn = gnn_class(**gnn.params)

        # Now load the full encoder+gnn model
        module = importlib.import_module(f"src.models.{model_type}")
        model_class = getattr(module, model_name)
        return model_class(encoder=encoder, gnn=gnn)

    else:
        # Simple model
        module = importlib.import_module(f"src.models.{model_type}")
        model_class = getattr(module, model_name)
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
