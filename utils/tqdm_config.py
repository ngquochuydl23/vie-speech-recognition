import yaml

class TQDMConfigs:
    def __init__(self):
        with open('configs.yaml', 'r') as file:
            config = yaml.safe_load(file)
            self.train_progressbar_color = config['tqdm_configs']['train_progressbar_color']
            self.download_progressbar_color = config['tqdm_configs']['download_progressbar_color']
            
    