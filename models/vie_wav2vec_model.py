from models.base_model import BaseModel


class VieWavToVecModel(BaseModel):
    def __init__(self, pretrained_path, special_tokens):
        super().__init__(pretrained_path, special_tokens)