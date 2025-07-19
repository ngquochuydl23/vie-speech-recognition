from ctypes import Union
from typing import Any
import torch
import os
from tqdm import tqdm
from torch.amp import autocast
from typing import Dict, Union
from time import sleep
from jiwer import wer
from utils.tqdm_config import TQDMConfigs
from logger import logging
class Trainer:
    def __init__(self,
                 resume,
                 preload,
                 epochs,
                 steps_per_epoch,

                 model,
                 compute_metric,
                 processor,
                 train_dl,
                 val_dl,
                 train_sampler,
                 val_sampler,

                 optimizer,
                 scheduler,
                 save_dir,

                 gradient_accumulation_steps,
                 use_amp,
                 max_clip_grad_norm,
                 sampling_rate=16000,
                 stateful_metrics=None,
                 save_max_metric_score=False,
                 ):

        self.resume = resume
        self.preload = preload
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.start_epoch = 0
        self.pbar_step = 0

        self.model = model
        self.compute_metric = compute_metric
        self.processor = processor
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.max_clip_grad_norm = max_clip_grad_norm

        self.use_distill = False
        self.use_amp = use_amp

        self.completed_steps = 0
        self.sampling_rate = sampling_rate
        self.stateful_metrics = stateful_metrics
        self.save_max_metric_score = save_max_metric_score
        self.best_score = None
        
        self.train_progressbar_color = TQDMConfigs().train_progressbar_color

    def get_grad_norm(self, params, scale=1) -> torch.tensor:
        """Compute grad norm given a gradient scale."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = (p.grad.detach().data / scale).norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _train_epoch(self, epoch) -> None:
        self.train_sampler.set_epoch(epoch)
        pbar = tqdm(total=self.steps_per_epoch, desc=f"Epoch {epoch + 1}", leave=False, ncols=100, colour=self.train_progressbar_color)

        for dl_step, batch in enumerate(self.train_dl):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device='cuda:0')
                    batch[k] = v.to('cuda:0')
            self.model.train()
            outputs = self.model(**batch)

            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / self.gradient_accumulation_steps / batch['input_values'].shape[0]
            self.scaler.scale(loss).backward()

            # Optimize step
            if (dl_step + 1) % self.gradient_accumulation_steps == 0 or dl_step == len(self.train_dl) - 1:
                # compute grad norm for monitoring
                grad_norm = self.get_grad_norm(self.model.parameters(), scale=self.scaler.get_scale())

                # gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)

                # update parameters
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                scale_after = self.scaler.get_scale()
                is_overflown = scale_after < scale_before
                if not is_overflown:
                    self.scheduler.step()
                    
                train_logs = {
                    "loss": loss * self.gradient_accumulation_steps,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "grad_norm": grad_norm,
                }
                train_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in train_logs.items()}
                self.pbar_step += 1
                self.completed_steps += 1
                sleep(1)
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

        logging.info("\nValidation is in progress...")
        self.model.eval()
        val_logs = self._valid_epoch(epoch)

        # print("\nSaving checkpoint...")

        # write val logs
        # self.writer.update(self.completed_steps, 'Validation', val_logs)
        # pbar.update(self.pbar_step + 1, "val_", val_logs)

        # Save best
        # if self._is_best_epoch(val_logs['wer'], save_max_metric_score=self.save_max_metric_score):
        #    self._save_checkpoint(epoch, is_best_epoch=True)
        # else:
        #    self._save_checkpoint(epoch, is_best_epoch=False)

    def _is_best_epoch(self, score, save_max_metric_score=True) -> bool:
        if save_max_metric_score and score >= self.best_score:
            self.best_score = score
            return True
        elif not save_max_metric_score and score <= self.best_score:
            self.best_score = score
            return True
        return False

    def _save_checkpoint(self, epoch: int, is_best_epoch: bool = False) -> None:
        logging.info(f"\n Saving model checkpoint...")

        state_dict = {
            "epoch": epoch, 
            "best_score": self.best_score, 
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(), 
            "scheduler": self.scheduler.state_dict(),
            "completed_steps": self.completed_steps, 
            "model": self.model.state_dict()
        }
        
        torch.save(state_dict, os.path.join(self.save_dir, "latest_model.tar"))
        torch.save(state_dict, os.path.join(self.save_dir, f"model_{str(epoch)}.tar"))

        # If the model get a best metric score (is_best_epoch=True) in the current epoch,
        # the model checkpoint will be saved as "best_model.tar."
        # The newer best-scored checkpoint will overwrite the older one.
        # if is_best_epoch:
        #    torch.save(state_dict, os.path.join(self.save_dir, "best_model.tar"))

    ##
    #   self.model.save_pretrained(self.config["huggingface"]["args"]["local_dir"])

    # if self.config["huggingface"]["push_to_hub"] and self.config["huggingface"]["push_every_validation_step"]:
    #     self._push_to_hub("update_best_model", True)

    def _valid_epoch(self, epoch) -> Dict[str, Union[Any, float]]:
        self.val_sampler.set_epoch(epoch)
        # init logs
        val_logs = {
            "loss": 0,
            "wer": 0
        }

        for batch in self.val_dl:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device='cuda:0')
                    batch[k] = v.to('cuda:0')
            with torch.no_grad():
                with autocast(enabled=self.use_amp, device_type='cuda'):
                    outputs = self.model(**batch)

            val_logs["loss"] += outputs.loss / len(self.val_dl)
            val_logs["wer"] += torch.tensor(self.compute_wer(outputs.logits, batch["labels"])) / len(self.val_dl)

        val_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in val_logs.items()}
        return val_logs

    def validate(self) -> None:
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self._valid_epoch(epoch)

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self._train_epoch(epoch)

    def compute_wer(self, logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
        return wer(label_str, pred_str)
