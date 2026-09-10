"""Shared configuration for the runnable Wi-Fi decoding experiments."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace


@dataclass
class Config:
    root_dir: str = os.environ.get("WIFI_DATA_ROOT", "")
    n_fft: int = 64
    n_symbols: int = 12
    num_bits: int = 24
    num_channels: int = 16
    num_res_blocks: int = 6
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15
    random_seed: int = 42
    num_workers: int = 4
    use_phase_augment: bool = True
    output_dir: str = os.environ.get("WIFI_OUTPUT_DIR", "outputs/deepmon")
    target_protocol: str = "Non-HT"

    @classmethod
    def from_cli(cls, default_output_dir: str | None = None) -> "Config":
        defaults = cls()
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data-root",
            default=defaults.root_dir,
            help="Dataset root containing Train/ and Test/ (or set WIFI_DATA_ROOT).",
        )
        parser.add_argument(
            "--output-dir",
            default=default_output_dir or defaults.output_dir,
            help="Directory for checkpoints and training artifacts.",
        )
        parser.add_argument("--epochs", type=int, default=defaults.num_epochs)
        parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
        parser.add_argument("--workers", type=int, default=defaults.num_workers)
        parser.add_argument("--seed", type=int, default=defaults.random_seed)
        args = parser.parse_args()
        return replace(
            defaults,
            root_dir=args.data_root,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.workers,
            random_seed=args.seed,
        )

    def validate_data_root(self) -> None:
        if not self.root_dir:
            raise FileNotFoundError(
                "Dataset path is not configured. Pass --data-root PATH or set "
                "WIFI_DATA_ROOT. The repository does not include the packet dataset."
            )
        missing = [
            split for split in ("Train", "Test")
            if not os.path.isdir(os.path.join(self.root_dir, split))
        ]
        if missing:
            raise FileNotFoundError(
                f"Dataset root {self.root_dir!r} is missing: {', '.join(missing)}"
            )
