from datetime import datetime


def log(msg):
    """Log message with timestamp to monitor training."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_run_name(config):
    """Generates a run name from the configuration file."""
    name = []

    model_type = config.model.type
    name.append(f"{model_type}")
    if model_type == "encoder_gnn":
        encoder = config.model.encoder
        gnn = config.model.gnn

        # Add encoder
        name.append(f"{encoder.type}")
        # Add model parameters
        for key, value in encoder.params.items():
            name.append(f"{key}_{value}")

        # Add GNN
        name.append(f"{gnn.type}")
        # Add model parameters
        for key, value in gnn.params.items():
            name.append(f"{key}_{value}")

    else:

        # Simple model
        model_params = config.model.params
        # Add model parameters
        for key, value in model_params.items():
            name.append(f"{key}_{value}")

    # Add all other parameters not related to model
    for key, value in config.items():
        if key != "model":
            name.append(f"{key}_{value}")

    return "_".join(name)
