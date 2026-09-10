import unittest

import numpy as np
import torch

from config import Config
from dataset import wave_to_2d_spectrogram
from model import DeepMonModel
from model_1d import DeepMonNoFFT
from model_attention import DeepMonAttention


class SignalPreprocessingTests(unittest.TestCase):
    def test_fft_shape_and_dtype(self):
        wave = np.ones(100, dtype=np.complex64)
        result = wave_to_2d_spectrogram(wave, n_fft=64, n_symbols=12)
        self.assertEqual(result.shape, (2, 12, 64))
        self.assertEqual(result.dtype, np.float32)

    def test_missing_data_root_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "--data-root"):
            Config(root_dir="").validate_data_root()


class ModelShapeTests(unittest.TestCase):
    def test_runnable_models_emit_24_logits(self):
        with torch.no_grad():
            self.assertEqual(DeepMonModel()(torch.randn(2, 2, 12, 64)).shape, (2, 24))
            self.assertEqual(DeepMonAttention()(torch.randn(2, 2, 12, 64)).shape, (2, 24))
            self.assertEqual(DeepMonNoFFT()(torch.randn(2, 2, 768)).shape, (2, 24))


if __name__ == "__main__":
    unittest.main()
