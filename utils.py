import yaml

def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

        
def format_metrics(metrics, split, prefix=""):
    log = f"[{split}]" + prefix
    log += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    return log