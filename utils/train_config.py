import yaml


class TrainConfigs:
    def __init__(self):
        with open("configs.yaml", "r") as file:
            config = yaml.safe_load(file)["train_configs"]
            self.config = {
                "sampling_rate": int(config["sampling_rate"]),
                "epochs": int(config["epochs"]),
                "save_dir": str(config["save_dir"]),
                "use_amp": bool(config["use_amp"]),
                "compute_metric": list(config["compute_metric"]),
                "gradient_accumulation_steps": int(
                    config["gradient_accumulation_steps"]
                ),
                "max_clip_grad_norm": float(config["max_clip_grad_norm"]),
                "n_gpus": int(config["n_gpus"]),
                "batch_size": int(config["batch_size"]),
                "n_workers": int(config["n_workers"]),
                "pretrained_path": str(config["pretrained_path"]),
                "pin_memory": bool(config["pin_memory"]),
                "prefetch_factor": int(config["prefetch_factor"]),
                "warmup_steps": int(config["warmup_steps"]),
                "lr": float(config["lr"]),
                "resume": config.get("resume"),
                "preload": config.get("preload"),
                "special_tokens": {
                    "bos_token": str(config["special_tokens"]["bos_token"]),
                    "eos_token": str(config["special_tokens"]["eos_token"]),
                    "unk_token": str(config["special_tokens"]["unk_token"]),
                    "pad_token": str(config["special_tokens"]["pad_token"]),
                },
                "data_dir": str(config["data_dir"]),
            }

    def get_config(self) -> dict:
        return self.config
