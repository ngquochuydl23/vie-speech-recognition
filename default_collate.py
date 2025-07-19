import torch
from typing import Dict


class DefaultCollate:
    def __init__(self, processor, sr) -> None:
        self.processor = processor
        self.sr = sr

    def __call__(self, inputs) -> Dict[str, torch.Tensor]:
        features, transcripts = zip(*inputs)
        features, transcripts = list(features), list(transcripts)
        batch = self.processor(
            features,
            sampling_rate=self.sr,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True
        )
        labels_batch = self.processor.tokenizer(
            transcripts,
            padding="longest",
            return_tensors="pt"
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id,
            -100
        )

        return batch
