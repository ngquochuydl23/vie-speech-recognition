import torch
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from logger import logging

class BaseModel:
    def __init__(self, pretrained_path, special_tokens):
        self.special_tokens = special_tokens
        self.pretrained_path = pretrained_path
        self.tokenizer = Wav2Vec2CTCTokenizer("vocab.json", **special_tokens, word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.pretrained_path)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_path,
            ctc_loss_reduction="sum",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
            gradient_checkpointing=False
        )

    def load_from_checkpoints(self, checkpoint_name="latest_model.tar"):
        checkpoint_path = os.path.join('./saved', checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            logging.warning(f"No checkpoint found at {checkpoint_path}. Skipping resume.")
            return

        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        self.model.load_state_dict(checkpoint['model'])
        logging.info(f"Successfully resumed from checkpoint '{checkpoint_name}'")
        return self.model, checkpoint


    def get_pretrained_model(self):
        return self.model
    
    
    def get_processor(self):
        return self.processor
