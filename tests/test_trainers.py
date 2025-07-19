import unittest
from unittest.mock import MagicMock, patch
import torch
import os

from ..trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.scheduler = MagicMock()
        self.scaler = MagicMock()
        self.processor = MagicMock()
        self.train_dl = MagicMock()
        self.val_dl = MagicMock()
        self.train_sampler = MagicMock()
        self.val_sampler = MagicMock()

        self.trainer = Trainer(
            resume=False,
            preload=False,
            epochs=1,
            steps_per_epoch=1,
            model=self.model,
            compute_metric=None,
            processor=self.processor,
            train_dl=self.train_dl,
            val_dl=self.val_dl,
            train_sampler=self.train_sampler,
            val_sampler=self.val_sampler,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_dir="checkpoints",
            gradient_accumulation_steps=1,
            use_amp=False,
            max_clip_grad_norm=1.0,
        )

    def test_is_best_epoch_min(self):
        self.trainer.best_score = 0.2
        # lower is better
        result = self.trainer._is_best_epoch(0.1, save_max_metric_score=False)
        self.assertTrue(result)
        self.assertEqual(self.trainer.best_score, 0.1)

    def test_is_best_epoch_max(self):
        self.trainer.best_score = 0.7
        # higher is better
        result = self.trainer._is_best_epoch(0.9, save_max_metric_score=True)
        self.assertTrue(result)
        self.assertEqual(self.trainer.best_score, 0.9)

    def test_get_grad_norm_with_grad(self):
        p1 = MagicMock()
        p1.grad = torch.tensor([3.0, 4.0])
        grad_norm = self.trainer.get_grad_norm([p1])
        self.assertAlmostEqual(grad_norm, 5.0)

    def test_get_grad_norm_without_grad(self):
        p1 = MagicMock()
        p1.grad = None
        grad_norm = self.trainer.get_grad_norm([p1])
        self.assertEqual(grad_norm, 0.0)

    @patch("torch.save")
    def test_save_checkpoint(self, mock_save):
        os.makedirs("checkpoints", exist_ok=True)
        self.trainer.best_score = 0.5
        self.trainer._save_checkpoint(0, is_best_epoch=True)
        self.assertEqual(mock_save.call_count, 3)  # latest_model, model_0, best_model

    def test_compute_wer(self):
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])  # shape (1, 2, 2)
        labels = torch.tensor([[1, 0]])
        self.processor.batch_decode.return_value = ["hello", "helo"]
        wer_score = self.trainer.compute_wer(logits, labels)
        self.assertGreaterEqual(wer_score, 0.0)
        self.assertLessEqual(wer_score, 1.0)


if __name__ == "__main__":
    unittest.main()
