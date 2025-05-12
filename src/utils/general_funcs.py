from datetime import datetime


def log(msg):
    """Log message with timestamp to monitor training."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_run_name(config):
    """Generates a run name from the configuration file."""
    name = []

    #
    name.append(f"{config.model.type}")
    model_params = config.model.params
    # Add model parameters
    for key, value in model_params.items():
        name.append(f"{key}_{value}")

    # Add all other parameters not related to model
    for key, value in config.items():
        if key != "model":
            name.append(f"{key}_{value}")

    return "_".join(name)
