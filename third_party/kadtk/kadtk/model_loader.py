import importlib.util
import logging
import math
import os
from abc import ABC, abstractmethod
from email.policy import strict
from pathlib import Path
from turtle import st
from typing import Literal, Optional, Union

import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from hypy_utils.downloader import download_file
from packaging import version
from torch import nn

from kadtk.models import msclap, panns

log = logging.getLogger(__name__)

class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, num_features: int, sr: int, audio_len: Optional[Union[float, int]] = None):
        self.audio_len = audio_len
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @torch.no_grad()
    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        if not embd.shape[-1] == self.num_features:
            raise RuntimeError(f"[{self.name}]: Expected {self.num_features} features, got {embd.shape[-1]}")
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd
    
    def postprocess_resoultion(self, audio: np.ndarray, emb: np.ndarray, pooling_resolution_sec: int = 1) -> np.ndarray:
        audio_dur = audio.shape[0] / self.sr
        pooling_resoultion = audio_dur / pooling_resolution_sec
        stride = int(emb.shape[0] / pooling_resoultion)
        emb = emb.unfold(0, stride, stride).mean(-1)
        return emb

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path) -> np.ndarray:
        wav_data, _ = soundfile.read(wav_file, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        
        # Ensure the audio length is correct
        if self.audio_len is not None and wav_data.shape[0] != int(self.audio_len * self.sr):
            raise RuntimeError(f"Audio length mismatch ({wav_data.shape[0] / self.sr:.2f} seconds != {self.audio_len} seconds)."
                                + f"\n\t- {wav_file}")
        return wav_data


class VGGishModel(ModelLoader):
    """
    S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
    """
    def __init__(self, use_pca: bool = False, use_activation: bool = False, audio_len: Optional[Union[float, int]] = None):
        super().__init__("vggish", 128, 16000, audio_len)
        self.use_pca = use_pca
        self.use_activation = use_activation

    def load_model(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        if not self.use_pca:
            self.model.postprocess = False
        if not self.use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        return self.model.forward(audio, self.sr)
    

class PANNsModel(ModelLoader):
    """
    Kong, Qiuqiang, et al., "Panns: Large-scale pretrained audio neural networks for audio pattern recognition.",
    IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

    Specify the model to use (cnn14-32k, cnn14-16k, wavegram-logmel).
    """
    def __init__(self, variant: Literal['cnn14-32k', 'cnn14-16k', 'wavegram-logmel'], audio_len: Optional[Union[float, int]] = None):
        super().__init__(f"panns-{variant}", 2048, 
                         sr=16000 if variant == 'cnn14-16k' else 32000, audio_len=audio_len)
        self.variant = variant

    def load_model(self):
        url_dict = {
            'cnn14-32k': "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth",
            'cnn14-16k': "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
            'wavegram-logmel': "https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth"
        }
        
        self.model_file = Path(__file__).parent / ".model-checkpoints" / url_dict[self.variant].split('/')[-1].replace("%3D", "=")

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url_dict[self.variant], self.model_file)
            
        features_list = ["2048", "logits"]

        if self.variant == 'cnn14-16k':
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )

        elif self.variant == 'cnn14-32k':
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )

        elif self.variant == 'wavegram-logmel':
            self.model = panns.Wavegram_Logmel_Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )

        else:
            raise ValueError(f"Unexpected variant of PANNs model: {self.variant}.")
        
        state_dict = torch.load(self.model_file, weights_only=False)
        self.model.load_state_dict(state_dict["model"])
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        audio = torch.from_numpy(audio).float().to(self.device)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        if 'cnn14' in self.variant:
            emb = self.model.forward(audio)["2048"]
        else:
            emb = self.model.forward(audio)["embedding"]
        return emb
    
class EncodecEmbModel(ModelLoader):
    """
    Encodec model from https://github.com/facebookresearch/encodec

    Thiss version uses the embedding outputs (continuous values of 128 features).
    """
    def __init__(self, variant: Literal['48k', '24k'] = '24k', audio_len: Optional[Union[float, int]] = None):
        super().__init__('encodec-emb' if variant == '24k' else f"encodec-emb-{variant}", 128,
                         sr=24000 if variant == '24k' else 48000, audio_len=audio_len)
        self.variant = variant

    def load_model(self):
        from encodec import EncodecModel
        if self.variant == '48k':
            self.model = EncodecModel.encodec_model_48khz()
            self.model.set_target_bandwidth(24)
        else:
            self.model = EncodecModel.encodec_model_24khz()
            self.model.set_target_bandwidth(12)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        segment_length = self.model.segment_length
        
        # The 24k model doesn't use segmenting
        if segment_length is None:
            return self._get_frame(audio)
        
        # The 48k model uses segmenting
        assert audio.dim() == 3
        _, channels, length = audio.shape
        assert channels > 0 and channels <= 2
        stride = segment_length

        encoded_frames: list[torch.Tensor] = []
        for offset in range(0, length, stride):
            frame = audio[:, :, offset:offset + segment_length]
            encoded_frames.append(self._get_frame(frame))

        # Concatenate
        encoded_frames = torch.cat(encoded_frames, dim=0) # [timeframes, 128]
        return encoded_frames

    def _get_frame(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            length = audio.shape[-1]
            duration = length / self.sr
            assert self.model.segment is None or duration <= 1e-5 + self.model.segment, f"Audio is too long ({duration} > {self.model.segment})"

            emb = self.model.encoder(audio.to(self.device)) # [1, 128, timeframes]
            emb = emb[0] # [128, timeframes]
            emb = emb.transpose(0, 1) # [timeframes, 128]
            emb = self.postprocess_resoultion(audio, emb)
            return emb
    
    def load_wav(self, wav_file: Path) -> np.ndarray:
        import torchaudio
        from encodec.utils import convert_audio

        wav, sr = torchaudio.load(str(wav_file))
        wav = convert_audio(wav, sr, self.sr, self.model.channels)

        # Ensure the audio length is correct
        if self.audio_len is not None and wav.shape[1] != int(self.audio_len * self.sr):
            raise RuntimeError(f"Audio length mismatch ({wav.shape[1] / self.sr:.2f} seconds != {self.audio_len} seconds)."
                                + f"\n\t- {wav_file}")
        # If it's longer than 3 minutes, cut it
        if wav.shape[1] > 3 * 60 * self.sr:
            wav = wav[:, :3 * 60 * self.sr]

        return wav.unsqueeze(0)
        
    def _decode_frame(self, emb: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            emb = torch.from_numpy(emb).float().to(self.device) # [timeframes, 128]
            emb = emb.transpose(0, 1) # [128, timeframes]
            emb = emb.unsqueeze(0) # [1, 128, timeframes]
            audio = self.model.decoder(emb) # [1, 1, timeframes]
            audio = audio[0, 0] # [timeframes]

            return audio.cpu().numpy()

    def postprocess_resoultion(self, audio: np.ndarray, emb: np.ndarray, pooling_resolution_sec: int = 1) -> np.ndarray:
        audio_dur = audio.shape[2] / self.sr
        pooling_resoultion = audio_dur / pooling_resolution_sec
        stride = int(emb.shape[0] / pooling_resoultion)
        emb = emb.unfold(0, stride, stride).mean(-1)
        return emb


class DACModel(ModelLoader):
    """
    DAC model from https://github.com/descriptinc/descript-audio-codec

    pip install descript-audio-codec
    """
    def __init__(self, audio_len: Optional[Union[float, int]] = None):
        self.sr = 44100
        super().__init__("dac-44kHz", 1024, self.sr, audio_len=audio_len)

    def load_model(self):
        from dac.utils import load_model
        self.model = load_model(tag='latest', model_type='44khz')
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio) -> torch.Tensor:
        import time

        from audiotools import AudioSignal

        audio: AudioSignal

        # Set variables
        win_len = 5.0
        overlap_hop_ratio = 0.5

        # Fix overlap window so that it's divisible by 4 in # of samples
        win_len = ((win_len * self.sr) // 4) * 4
        win_len = win_len / self.sr
        hop_len = win_len * overlap_hop_ratio

        stime = time.time()

        # Sanitize input
        audio.normalize(-16)
        audio.ensure_max_of_audio()

        nb, nac, nt = audio.audio_data.shape
        audio.audio_data = audio.audio_data.reshape(nb * nac, 1, nt)

        pad_length = math.ceil(audio.signal_duration / win_len) * win_len
        audio.zero_pad_to(int(pad_length * self.sr))
        audio = audio.collect_windows(win_len, hop_len)

        print(win_len, hop_len, audio.batch_size, f"(processed in {(time.time() - stime) * 1000:.0f}ms)")
        stime = time.time()

        emb = []
        for i in range(audio.batch_size):
            signal_from_batch = AudioSignal(audio.audio_data[i, ...], self.sr)
            signal_from_batch.to(self.device)
            e1 = self.model.encoder(signal_from_batch.audio_data).cpu() # [1, 1024, timeframes]
            e1 = e1[0] # [1024, timeframes]
            e1 = e1.transpose(0, 1) # [timeframes, 1024]
            emb.append(e1)

        emb = torch.cat(emb, dim=0)
        emb = self.postprocess_resoultion(audio, emb)
        return emb

    def load_wav(self, wav_file: Path) -> np.ndarray:
        from audiotools import AudioSignal
        wav = AudioSignal(wav_file)
        
        # Ensure the audio length is correct
        if self.audio_len is not None and wav.signal_duration != self.audio_len:
            raise RuntimeError(f"Audio length mismatch ({wav.signal_duration} seconds != {self.audio_len} seconds)."
                                + f"\n\t- {wav_file}")
        
        return wav

    def postprocess_resoultion(self, audio: np.ndarray, emb: np.ndarray, pooling_resolution_sec: int = 1) -> np.ndarray:
        audio_dur = audio.shape[2] / self.sr
        pooling_resoultion = audio_dur / pooling_resolution_sec
        stride = int(emb.shape[0] / pooling_resoultion)
        emb = emb.unfold(0, stride, stride).mean(-1)
        return emb


class MERTModel(ModelLoader):
    """
    MERT model from https://huggingface.co/m-a-p/MERT-v1-330M

    Please specify the layer to use (1-12).
    """
    def __init__(self, size='v1-95M', layer=12, limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        super().__init__(f"MERT-{size}" + ("" if layer == 12 else f"-{layer}"), 768, 24000, audio_len=audio_len)
        self.huggingface_id = f"m-a-p/MERT-{size}"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr
        
    def load_model(self):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        
        self.model = AutoModel.from_pretrained(self.huggingface_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.huggingface_id, trust_remote_code=True)
        # self.sr = self.processor.sampling_rate
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to 9 minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze() # [13 layers, timeframes, 768]
            out = out[self.layer] # [timeframes, 768]
            out = self.postprocess_resoultion(audio, out)
        return out


class CLAPLaionModel(ModelLoader):
    """
    CLAP model from https://github.com/LAION-AI/CLAP
    """
    
    def __init__(self, type: Literal['audio', 'music'], audio_len: Optional[Union[float, int]] = None):
        super().__init__(f"clap-laion-{type}", 512, 48000, audio_len=audio_len)
        self.type = type

        if type == 'audio':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt'
        elif type == 'music':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'

        self.model_file = Path(__file__).parent / ".model-checkpoints" / url.split('/')[-1]

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)
            
        # Patch the model file to remove position_ids (will raise an error otherwise)
        from importlib.metadata import version as metaversion

        import laion_clap
        import transformers
        if version.parse(transformers.__version__) >= version.parse("4.31.0") \
            and version.parse(metaversion("laion_clap")) < version.parse("1.1.6"): 
            self.patch_model_430(self.model_file)

    def patch_model_430(self, file: Path):
        """
        Patch the model file to remove position_ids (will raise an error otherwise)
        This is a new issue after the transformers 4.30.0 update
        Please refer to https://github.com/LAION-AI/CLAP/issues/127 and https://github.com/LAION-AI/CLAP/pull/118
        """
        # Create a "patched" file when patching is done
        patched = file.parent / f"{file.name}.patched.430"
        if patched.exists():
            return
        
        OFFENDING_KEY = "module.text_branch.embeddings.position_ids"
        log.warning("Patching LAION-CLAP's model checkpoints")
        
        # Load the checkpoint from the given path
        checkpoint = torch.load(file, map_location="cpu", weights_only=False)

        # Extract the state_dict from the checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Delete the specific key from the state_dict
        if OFFENDING_KEY in state_dict:
            del state_dict[OFFENDING_KEY]

        # Save the modified state_dict back to the checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint["state_dict"] = state_dict

        # Save the modified checkpoint
        torch.save(checkpoint, file)
        log.warning(f"Saved patched checkpoint to {file}")
        
        # Create a "patched" file when patching is done
        patched.touch()
        
    def load_model(self):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny' if self.type == 'audio' else 'HTSAT-base')
        self.model.load_ckpt(self.model_file)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        audio = audio.reshape(1, -1)

        # The int16-float32 conversion is used for quantization
        audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 10s chunks with 1s hop
        chunk_size = 10 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.get_audio_embedding_from_data(x = chunk, use_tensor=True)
                embeddings.append(emb)

        # Concatenate the embeddings
        emb = torch.cat(embeddings, dim=0) # [timeframes, 512]
        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)


class CdpamModel(ModelLoader):
    """
    CDPAM model from https://github.com/pranaymanocha/PerceptualAudio/tree/master/cdpam
    """
    def __init__(self, mode: Literal['acoustic', 'content'], audio_len: Optional[Union[float, int]] = None) -> None:
        super().__init__(f"cdpam-{mode}", 512, 22050, audio_len=audio_len)
        self.mode = mode
        assert mode in ['acoustic', 'content'], "Mode must be 'acoustic' or 'content'"

    def load_model(self):
        from cdpam import CDPAM
        self.model = CDPAM(dev=self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        audio = torch.from_numpy(audio).float().to(self.device)

        # Take 1s chunks
        chunk_size = self.sr
        frames = []
        for i in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, i:i+chunk_size]
            _, acoustic, content = self.model.model.base_encoder.forward(chunk.unsqueeze(1))
            v = acoustic if self.mode == 'acoustic' else content
            v = F.normalize(v, dim=1)
            frames.append(v)

        # Concatenate the embeddings
        emb = torch.cat(frames, dim=0) # [timeframes, 512]
        return emb

    def load_wav(self, wav_file: Path) -> np.ndarray:
        x, _  = librosa.load(wav_file, sr=self.sr)
        
        # Ensure the audio length is correct
        if self.audio_len is not None and x.shape[-1] != int(self.audio_len * self.sr):
            raise RuntimeError(f"Audio length mismatch ({x.shape[-1] / self.sr:.2f} seconds != {self.audio_len} seconds)."
                                + f"\n\t- {wav_file}")
        
        # Convert to 16 bit floating point
        x = np.round(x.astype(np.float) * 32768)
        x  = np.reshape(x, [-1, 1])
        x = np.reshape(x, [1, x.shape[0]])
        x  = np.float32(x)
        
        return x


class CLAPModel(ModelLoader):
    """
    CLAP model from https://github.com/microsoft/CLAP
    """
    def __init__(self, type: Literal['2023'], audio_len: Optional[Union[float, int]] = None):
        super().__init__(f"clap-{type}", 1024, 44100, audio_len=audio_len)
        self.type = type

        if type == '2023':
            url = 'https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth'

        self.model_file = Path(__file__).parent / ".model-checkpoints" / url.split('/')[-1]

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)

    def load_model(self):
        self.model = msclap.CLAP(self.model_file, version = self.type, use_cuda=self.device == torch.device('cuda'))
        #self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        audio = audio.reshape(1, -1)

        # The int16-float32 conversion is used for quantization
        #audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 7s chunks with 1s hop
        chunk_size = 7 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # zero-pad chunks to make equal length
        clen = [x.shape[1] for x in chunks]
        chunks = [np.pad(ch, ((0,0), (0,np.max(clen) - ch.shape[1]))) for ch in chunks]

        self.model.default_collate(chunks)

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.clap.audio_encoder(chunk)[0]
                embeddings.append(emb)

        # Concatenate the embeddings
        emb = torch.cat(embeddings, dim=0) # [timeframes, 1024]
        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)


class W2V2Model(ModelLoader):
    """
    W2V2 model from https://huggingface.co/facebook/wav2vec2-base-960h, https://huggingface.co/facebook/wav2vec2-large-960h

    Please specify the size ('base' or 'large') and the layer to use (1-12 for 'base' or 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'large'], layer: Literal['12', '24'], limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        model_dim = 768 if size == 'base' else 1024
        model_identifier = f"w2v2-{size}" + ("" if (layer == 12 and size == 'base') or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"facebook/wav2vec2-{size}-960h"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, Wav2Vec2Model
        
        self.model = Wav2Vec2Model.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]
            out = self.postprocess_resoultion(audio, out)

        return out


class HuBERTModel(ModelLoader):
    """
    HuBERT model from https://huggingface.co/facebook/hubert-base-ls960, https://huggingface.co/facebook/hubert-large-ls960

    Please specify the size ('base' or 'large') and the layer to use (1-12 for 'base' or 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'large'], layer: Literal['12', '24'], limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        model_dim = 768 if size == 'base' else 1024
        model_identifier = f"hubert-{size}" + ("" if (layer == 12 and size == 'base') or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"facebook/hubert-{size}-ls960"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, HubertModel

        self.model = HubertModel.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]
            out = self.postprocess_resoultion(audio, out)

        return out


class WavLMModel(ModelLoader):
    """
    WavLM model from https://huggingface.co/microsoft/wavlm-base, https://huggingface.co/microsoft/wavlm-base-plus, https://huggingface.co/microsoft/wavlm-large

    Please specify the model size ('base', 'base-plus', or 'large') and the layer to use (1-12 for 'base' or 'base-plus' and 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'base-plus', 'large'], layer: Literal['12', '24'], limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        model_dim = 768 if size in ['base', 'base-plus'] else 1024
        model_identifier = f"wavlm-{size}" + ("" if (layer == 12 and size in ['base', 'base-plus']) or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"patrickvonplaten/wavlm-libri-clean-100h-{size}"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, WavLMModel

        self.model = WavLMModel.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]
            out = self.postprocess_resoultion(audio, out)

        return out


class WhisperModel(ModelLoader):
    """
    Whisper model from https://huggingface.co/openai/whisper-base
    
    Please specify the model size ('tiny', 'base', 'small', 'medium', or 'large').
    """
    def __init__(self, size: Literal['tiny', 'base', 'small', 'medium', 'large'], audio_len: Optional[Union[float, int]] = None):
        dimensions = {
            'tiny': 384,
            'base': 512,
            'small': 768,
            'medium': 1024,
            'large': 1280
        }
        model_dim = dimensions.get(size)
        model_identifier = f"whisper-{size}"

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"openai/whisper-{size}"
        
    def load_model(self):
        from transformers import AutoFeatureExtractor, WhisperModel
        
        self.model = WhisperModel.from_pretrained(self.huggingface_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        inputs = self.feature_extractor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        input_features = inputs.input_features
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state # [1, timeframes, 512]
            out = out.squeeze() # [timeframes, 384 or 512 or 768 or 1024 or 1280]

        return out
    
    
class OpenL3Model(ModelLoader):
    """
    OpenL3 model from https://github.com/marl/openl3
    We only use 512-dimensional embeddings, not the 6144-dimensional embeddings.
    Reference: https://github.com/Stability-AI/stable-audio-metrics
    """
    def __init__(self, variant: Literal['mel256-env', 'mel256-music', 'mel128-env', 'mel128-music'], audio_len: Optional[Union[float, int]] = None):
        super().__init__(f"openl3-{variant}", 512, 48000, audio_len=audio_len)
        self.input_repr, self.content_type = variant.split('-')
        self.HOP_SIZE = 0.5  # openl3 hop_size in seconds (openl3 window is 1 sec)

    def _download_weight(self):
        import gzip
        
        weight_filename = f'openl3_audio_{self.input_repr}_{self.content_type}.h5'
        base_url = 'https://github.com/marl/openl3/raw/models/'
        model_version_str = 'v0_4_0'
        
        self.model_file = Path(__file__).parent / ".model-checkpoints" / weight_filename

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            compressed_filename = f'{weight_filename.split(".h5")[0]}-{model_version_str}.h5.gz'
            compressed_path = self.model_file.parent / compressed_filename
            if not compressed_path.exists():
                download_file(base_url + compressed_filename, compressed_path)
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(self.model_file, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        # Check if the file is downloaded correctly
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_file}")
        
    
    def load_model(self):
        from .models.openl3 import load_audio_embedding_model_from_path
        
        self._download_weight()
        
        self.model = load_audio_embedding_model_from_path(
            model_path=self.model_file, input_repr=self.input_repr,
            embedding_size=self.num_features, frontend='kapre'
        )
        # self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        from .models.openl3 import get_audio_embedding
        
        embs, _ = get_audio_embedding(audio, self.sr, model=self.model, verbose=False, hop_size=self.HOP_SIZE) # embeddings, timestamps
        
        return torch.from_numpy(embs) # embs: [timeframes, 512]

    def load_wav(self, wav_file: Path) -> np.ndarray:
        # mono-channel audio only (refer to FrechetAudioDistance.load_audio)
        wav_data, sr = librosa.load(wav_file, sr=None)
        assert sr == self.sr, f"Sample rate mismatch: {sr} from file != {self.sr} from model"
        
        if wav_data.shape[0] < sr: 
            print('Audio shorter than 1 sec, openl3 will zero-pad it:', wav_file)
            # Ensure the audio length is correct
        if self.audio_len is not None and wav_data.shape[0] != int(self.audio_len * self.sr):
            raise RuntimeError(f"Audio length mismatch ({wav_data.shape[0] / self.sr:.2f} seconds != {self.audio_len} seconds)."
                                + f"\n\t- {wav_file}")
        
        return wav_data
    
    
class PaSSTModel(ModelLoader):
    """
    PassT model from https://github.com/kkoutini/passt_hear21
    Base models are trained on AudioSet.
    """
    def __init__(self, variant: Literal['base-10s', 'base-20s', 'base-30s', 'openmic', 'fsd50k'], audio_len: Optional[Union[float, int]] = None):
        self.EMBED_DIM = 768
        dimensions = {
            'base-10s': self.EMBED_DIM+527, # 1295
            'base-20s': self.EMBED_DIM+527,
            'base-30s': self.EMBED_DIM+527,
            'openmic': self.EMBED_DIM+20, # 788
            'fsd50k': self.EMBED_DIM+200 # 968
        }
        self.variant = variant
        super().__init__(f"passt-{variant}", dimensions[variant], 32000, audio_len=audio_len)
        self.limit = int(variant.split('-')[1].replace('s', '')) * self.sr if 'base' in variant else 10 * self.sr

    def load_model(self):
        from hear21passt.base import get_basic_model, get_model_passt
        
        self.model = get_basic_model(mode="all") # all: embeddings + logits
        if self.variant == 'base-20s':
            self.model.net = get_model_passt('passt_20sec', input_tdim=2000)
        elif self.variant == 'base-30s':
            self.model.net = get_model_passt('passt_30sec', input_tdim=3000)
        elif not self.variant == 'base-10s':
            self.model.net = get_model_passt(arch=self.variant, n_classes=self.num_features-self.EMBED_DIM)
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        from hear21passt.base import get_scene_embeddings
        
        audio = torch.from_numpy(audio).float().to(self.device)
        embs = get_scene_embeddings(audio, self.model)
        
        return embs # embs logit: [1, dimensions[variant]]

    def load_wav(self, wav_file: Path) -> np.ndarray:
        import torchaudio
        
        wav_data, _ = torchaudio.load(wav_file) # channel(=1) x time samples
        
        if self.audio_len is not None and wav_data.shape[-1] != int(self.audio_len * self.sr):
            raise RuntimeError(f"Audio length mismatch ({wav_data.shape[-1] / self.sr:.2f} seconds != {self.audio_len} seconds)."
                                + f"\n\t- {wav_file}")
        
        if wav_data.shape[-1] < self.limit:
            wav_data = torch.nn.functional.pad(wav_data, (0, self.limit - wav_data.shape[-1]), mode='constant', value=0.0)
        wav_data = wav_data[:, :self.limit].numpy()
        
        return wav_data
    

def get_all_models(audio_len: Optional[Union[float, int]] = None) -> list[ModelLoader]:
    """
    Returns a list of all available models.
    
    Parameters:
    - audio_len: The length of the audio in seconds. 
                If the audio does not match this length, it will raise an error.
                If None(default), it will not check the length.
    
    Returns:
    - A list of all available models.
    """
    ms = [
        CLAPModel('2023', audio_len=audio_len),
        CLAPLaionModel('audio', audio_len=audio_len), CLAPLaionModel('music', audio_len=audio_len),
        VGGishModel(audio_len=audio_len), 
        PANNsModel('cnn14-32k',audio_len=audio_len), PANNsModel('cnn14-16k',audio_len=audio_len),
        PANNsModel('wavegram-logmel',audio_len=audio_len),
        *(MERTModel(layer=v, audio_len=audio_len) for v in range(1, 13)),
        EncodecEmbModel('24k', audio_len=audio_len), EncodecEmbModel('48k', audio_len=audio_len), 
        DACModel(audio_len=audio_len),
        CdpamModel('acoustic', audio_len=audio_len), CdpamModel('content', audio_len=audio_len),
        *(W2V2Model('base', layer=v, audio_len=audio_len) for v in range(1, 13)),
        *(W2V2Model('large', layer=v, audio_len=audio_len) for v in range(1, 25)),
        *(HuBERTModel('base', layer=v, audio_len=audio_len) for v in range(1, 13)),
        *(HuBERTModel('large', layer=v, audio_len=audio_len) for v in range(1, 25)),
        *(WavLMModel('base', layer=v, audio_len=audio_len) for v in range(1, 13)),
        *(WavLMModel('base-plus', layer=v, audio_len=audio_len) for v in range(1, 13)),
        *(WavLMModel('large', layer=v, audio_len=audio_len) for v in range(1, 25)),
        WhisperModel('tiny', audio_len=audio_len), WhisperModel('small', audio_len=audio_len),
        WhisperModel('base', audio_len=audio_len), WhisperModel('medium', audio_len=audio_len),
        WhisperModel('large', audio_len=audio_len),
        OpenL3Model('mel256-env', audio_len=audio_len), OpenL3Model('mel256-music', audio_len=audio_len),
        OpenL3Model('mel128-env', audio_len=audio_len), OpenL3Model('mel128-music', audio_len=audio_len),
        PaSSTModel('base-10s', audio_len=audio_len), PaSSTModel('base-20s', audio_len=audio_len),
        PaSSTModel('base-30s', audio_len=audio_len), PaSSTModel('openmic', audio_len=audio_len),
        PaSSTModel('fsd50k', audio_len=audio_len)
    ]
    if importlib.util.find_spec("dac") is not None:
        ms.append(DACModel())
    if importlib.util.find_spec("cdpam") is not None:
        ms += [CdpamModel('acoustic'), CdpamModel('content')]

    return ms
