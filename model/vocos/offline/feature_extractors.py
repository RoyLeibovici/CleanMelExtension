from typing import List

import torch
import librosa
from encodec import EncodecModel
from torch import nn
from torch import Tensor
from typing import Optional
from torchaudio.transforms import Spectrogram, MelScale
from model.vocos.offline.modules import safe_log


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class LibrosaMelScale(nn.Module):
    r"""This MelScale has a create_fb_matrix function that can be used to create a filterbank matrix.
    same as previous torchaudio version
    """
    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super(LibrosaMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))
        _mel_options = dict(
            sr=sample_rate,
            n_fft=(n_stft - 1) * 2,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
            htk=mel_scale=="htk",
            norm=norm
        )
        fb = torch.from_numpy(librosa.filters.mel(**_mel_options).T).float()
        self.register_buffer("fb", fb)
    
    def forward(self, specgram):
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_win: int,
        n_hop: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        power: int,
        center: bool,
        normalize: bool,
        onesided: bool,
        mel_norm: str | None,
        mel_scale: str,
        librosa_mel: bool = True,
        clip_val: float = 1e-7
        ):
        super().__init__()
        # This implementation vs torchaudio.transforms.MelSpectrogram: Add librosa melscale
        # librosa melscale is numerically different from the torchaudio melscale (x_diff > 1e-5)
        self.n_fft = n_fft   
        self.spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=n_win,
            hop_length=n_hop,
            power=power,
            normalized=normalize,
            center=center,
            onesided=onesided,
        )
        mel_method = LibrosaMelScale if librosa_mel else MelScale
        self.mel_scale = mel_method(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm=mel_norm,
            mel_scale=mel_scale,
        )
        self.clip_val = clip_val
        
    def forward(self, x: Tensor) -> Tensor:
        # Compute Spectrogram
        specgram = self.spectrogram(x)
        mel_specgram = self.mel_scale(specgram)
        return safe_log(mel_specgram, self.clip_val)

class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)
