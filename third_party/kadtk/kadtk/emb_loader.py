import multiprocessing
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from hypy_utils.tqdm_utils import tmap, tq

from kadtk.model_loader import ModelLoader
from kadtk.utils import find_sox_formats, get_cache_embedding_path

sox_path = os.environ.get('SOX_PATH', 'sox')
ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')
TORCHAUDIO_RESAMPLING = True

if not(TORCHAUDIO_RESAMPLING):
    if not shutil.which(sox_path):
        raise Exception(f"Could not find SoX executable at {sox_path}, please install SoX and set the SOX_PATH environment variable.")
    if not shutil.which(ffmpeg_path):
        raise Exception(f"Could not find ffmpeg executable at {ffmpeg_path}, please install ffmpeg and set the FFMPEG_PATH environment variable.")

class EmbeddingLoader:
    def __init__(self, model: ModelLoader, audio_load_worker: int = 8, load_model: bool = True):
        self.ml = model
        self.audio_load_worker = audio_load_worker
        self.sox_formats = find_sox_formats(sox_path)
        if load_model:
            self.ml.load_model()
            self.loaded = True

    def load_audio(self, f: Union[str, Path]):
        f = Path(f)

        # Create a directory for storing normalized audio files
        cache_dir = f.parent / "convert" / str(self.ml.sr)
        new = (cache_dir / f.name).with_suffix(".wav")

        if not new.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            if TORCHAUDIO_RESAMPLING:
                x, fsorig = torchaudio.load(f)
                x = torch.mean(x,0).unsqueeze(0) # convert to mono
                resampler = torchaudio.transforms.Resample(
                    fsorig,
                    self.ml.sr,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
                y = resampler(x)
                torchaudio.save(new, y, self.ml.sr, encoding="PCM_S", bits_per_sample=16)
            else:                
                sox_args = ['-r', str(self.ml.sr), '-c', '1', '-b', '16']
    
                # ffmpeg has bad resampling compared to SoX
                # SoX has bad format support compared to ffmpeg
                # If the file format is not supported by SoX, use ffmpeg to convert it to wav
    
                if f.suffix[1:] not in self.sox_formats:
                    # Use ffmpeg for format conversion and then pipe to sox for resampling
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp) / 'temp.wav'
    
                        # Open ffmpeg process for format conversion
                        subprocess.run([
                            ffmpeg_path, 
                            "-hide_banner", "-loglevel", "error", 
                            "-i", f, tmp])
                        
                        # Open sox process for resampling, taking input from ffmpeg's output
                        subprocess.run([sox_path, tmp, *sox_args, new])
                        
                else:
                    # Use sox for resampling
                    subprocess.run([sox_path, f, *sox_args, new])

        return self.ml.load_wav(new)
    
    def read_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Read embedding from a cached file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)
        if not cache.exists():
            raise ValueError(f"Embedding file {cache} does not exist.")
        emb = np.load(cache)
        return emb

    def load_embeddings(self, dir: Union[str, Path], max_count: int = -1, concat: bool = True):
        """
        Load embeddings for all audio files in a directory.
        """
        files = list(Path(dir).glob("*.*"))
        print(f"Loading {len(files)} audio files from {dir}...")

        return self._load_embeddings(files, max_count=max_count, concat=concat)

    def _load_embeddings(self, files: list[Path], max_count: int = -1, concat: bool = True):
        """
        Load embeddings for a list of audio files.
        """
        if len(files) == 0:
            raise ValueError("No files provided")

        # Load embeddings
        if max_count == -1:
            embd_lst = tmap(self.read_embedding_file, files, desc="Loading audio files...", max_workers=self.audio_load_worker)
        else:
            total_len = 0
            embd_lst = []
            for f in tq(files, "Loading files"):
                embd_lst.append(self.read_embedding_file(f))
                total_len += embd_lst[-1].shape[0]
                if total_len > max_count:
                    break
                
        # Concatenate embeddings if needed
        if concat:
            return np.concatenate(embd_lst, axis=0)
        else:
            return embd_lst, files

    def cache_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)

        if cache.exists():
            return

        # Load file, get embedding, save embedding
        wav_data = self.load_audio(audio_dir)
        embd = self.ml.get_embedding(wav_data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)

# Main
def _cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml, kwargs = args
    emb_loader = EmbeddingLoader(ml, **kwargs)
    for f in fs:
        print(f"Loading {f} using {ml.name}")
        emb_loader.cache_embedding_file(f)


def cache_embedding_files(files: Union[list[Path], str, Path], ml: ModelLoader, workers: int = 8, 
                          force_emb_encode: bool = False, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    Params:
    - files (list[Path] | str | Path): List of audio files or a directory containing audio files.
    - ml (ModelLoader): ModelLoader instance to use.
    - workers (int): Number of workers to use.
    - force_emb_encode (bool): If True, re-extract embeddings even if they already exist.
    """
    if isinstance(files, (str, Path)):
        files = list(Path(files).glob('*.*'))

    if force_emb_encode:
        emb_path = files[0].parent / "embeddings" / ml.name
        if os.path.exists(emb_path):
            # Remove the folder and its contents
            shutil.rmtree(emb_path)
            print(f"The folder '{emb_path}' has been successfully removed.")
        

    # Filter out files that already have embeddings
    files = [f for f in files if not get_cache_embedding_path(ml.name, f).exists()]

    if len(files) == 0:
        print("All files already have embeddings, skipping.")
        return

    print(f"Loading {len(files)} audio files...")

    # Split files into batches
    batches = list(np.array_split(files, workers))
    
    # Cache embeddings in parallel
    multiprocessing.set_start_method('spawn', force=True)
    with torch.multiprocessing.Pool(workers) as pool:
        pool.map(_cache_embedding_batch, [(b, ml, kwargs) for b in batches])
        