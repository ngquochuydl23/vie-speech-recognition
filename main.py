from torch.utils.data import DataLoader

from datasets.combined_dataset import CombinedDatasetFactory
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import torch
import os
from default_collate import DefaultCollate
from trainer import Trainer
from warmup_lr import WarmupLR
from utils.train_config import TrainConfigs
from logger import logging
import traceback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == '__main__':
    clear_console()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
        
    train_configs = TrainConfigs().get_config()
    
    combined_dataset_factory = CombinedDatasetFactory(data_dir=train_configs["data_dir"])
    combined_dataset_factory.save_vocab_dict(train_configs["special_tokens"], "vocab.json")

    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", **train_configs["special_tokens"], word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(train_configs["pretrained_path"])
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    default_collate = DefaultCollate(processor, train_configs["sampling_rate"])

    train_ds, val_ds, test_ds = combined_dataset_factory.get_dataset(split=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=train_configs["n_gpus"],
        rank=0,
        shuffle=True,
        drop_last=True
    )

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=train_configs["batch_size"],
        num_workers=train_configs["n_workers"],
        pin_memory=train_configs["pin_memory"],
        drop_last=True,
        sampler=train_sampler,
        collate_fn=default_collate,
        prefetch_factor=train_configs["prefetch_factor"],
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=train_configs["n_gpus"],
        rank=0,
        shuffle=True,
        drop_last=True
    )

    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=train_configs["batch_size"],
        num_workers=train_configs["n_workers"],
        pin_memory=train_configs["pin_memory"],
        drop_last=True,
        sampler=val_sampler,
        collate_fn=default_collate,
        prefetch_factor=train_configs["prefetch_factor"],
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        train_configs["pretrained_path"],
        ctc_loss_reduction="sum",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        gradient_checkpointing=False
    )
    model.freeze_feature_encoder()
    model.to('cuda:0')

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=train_configs["lr"])
    steps_per_epoch = (len(train_dl) // train_configs["gradient_accumulation_steps"]) + (
            len(train_dl) % train_configs["gradient_accumulation_steps"] != 0)
    
    scheduler = WarmupLR(optimizer=optimizer, warmup_steps=train_configs["warmup_steps"])
    trainer = Trainer(
        resume=train_configs["resume"],
        preload=train_configs["preload"],
        epochs=train_configs["epochs"],
        steps_per_epoch=steps_per_epoch,
        model=model,
        compute_metric=train_configs["compute_metric"],
        processor=processor,
        train_dl=train_dl,
        val_dl=val_dl,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=train_configs["save_dir"],
        gradient_accumulation_steps=train_configs["gradient_accumulation_steps"],
        use_amp=train_configs["use_amp"],
        max_clip_grad_norm=train_configs["max_clip_grad_norm"],
        sampling_rate=train_configs["sampling_rate"],
        stateful_metrics=None,
        save_max_metric_score=False,
    )
    
    try:
        clear_console()
        trainer.train()   
    except Exception as e:
        logging.error("An error occurred during training:")
        logging.error(traceback.format_exc())     

