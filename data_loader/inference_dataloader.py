import torch
import numpy as np
import soundfile as sf
from glob import glob
from typing import Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from data_loader.utils.collate_func import default_collate_func
from data_loader.utils.my_distributed_sampler import MyDistributedSampler


class InferenceDataset(Dataset):
    def __init__(self, speech_dir: str, sample_rate: int) -> None:
        super().__init__()
        self.speech_dir = speech_dir
        self.uttrs = glob(f"{self.speech_dir}/**/*.wav", recursive=True) + glob(f"{self.speech_dir}/**/*.flac", recursive=True)
        self.uttrs.sort()
        
        # sanity check
        for uttr in self.uttrs:
            assert sf.info(uttr).samplerate == sample_rate, f"{uttr} has wrong sample rate"
        
    def __getitem__(self, index_seed: tuple[int, int]):
        uttr_id = index_seed[0]
        wavename = self.uttrs[uttr_id]
        noisy, _ = sf.read(wavename)
        noisy = noisy.astype(np.float32)
        if noisy.ndim > 1:
            noisy = noisy[:, 0]    # multi-channel to single channel
        return torch.tensor(noisy), wavename

    def __len__(self):
        return len(self.uttrs)

class InferenceDataModule(LightningDataModule):

    def __init__(
        self,
        speech_dir: str = './src/demos/',  # a dir contains [train-clean-100, train-clean-360]
        batch_size: int = 1,
        num_workers: int = 2,
        collate_func: Callable = default_collate_func,
        seed: int = 2,  # random seeds for train, val and test sets
        pin_memory: bool = True,
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
        sample_rate=16000,
    ):
        super().__init__()
        self.speech_dir = speech_dir
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_func = collate_func
        self.seed = seed
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.sample_rate = sample_rate

    def construct_dataloader(self):
        ds = InferenceDataset(speech_dir=self.speech_dir, sample_rate=self.sample_rate)
        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=self.seed, shuffle=False),  #
            batch_size=self.batch_size,  #
            collate_fn=self.collate_func,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader()
