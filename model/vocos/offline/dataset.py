from dataclasses import dataclass

import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torchaudio
import warnings
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def customize_soxnorm(self, wav, gain=-3, factor=None):
            wav = np.clip(wav, a_max=1, a_min=-1)
            if factor is None:
                linear_gain = 10 ** (gain / 20)
                wav = wav / np.abs(wav).max() * linear_gain
                return wav,  linear_gain / np.abs(wav).max()
            else:
                wav = wav * factor
                return wav, None

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        try:
            y, sr = torchaudio.load(audio_path)
        except:
            warnings.warn(f"Error loading {audio_path}")
            return self.__getitem__(np.random.randint(len(self.filelist)))   
        if y.size(-1) == 0:
            return self.__getitem__(np.random.randint(len(self.filelist)))
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = self.customize_soxnorm(y, gain)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]
