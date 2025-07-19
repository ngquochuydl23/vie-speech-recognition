import yaml

class TrainConfigs:
    def __init__(self):
        with open('configs.yaml', 'r') as file:
            self.config = yaml.safe_load(file)['train_configs']
        
    def get_config(self) -> dict:
        return self.config

    