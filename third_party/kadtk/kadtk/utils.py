import subprocess
from pathlib import Path
from typing import Union

import numpy as np
from hypy_utils.nlp_utils import substr_between
from hypy_utils.tqdm_utils import pmap

PathLike = Union[str, Path]
    

def find_sox_formats(sox_path: str) -> list[str]:
    """
    Find a list of file formats supported by SoX
    """
    try:
        out = subprocess.check_output((sox_path, "-h")).decode()
        return substr_between(out, "AUDIO FILE FORMATS: ", "\n").split()
    except:
        return []


def get_cache_embedding_path(model: str, audio_dir: PathLike) -> Path:
    """
    Get the path to the cached embedding npy file for an audio file.

    :param model: The name of the model
    :param audio_dir: The path to the audio file
    """
    audio_dir = Path(audio_dir)
    return audio_dir.parent / "embeddings" / model / audio_dir.with_suffix(".npy").name

def get_cache_embedding_paths(model: str, audio_dir: PathLike) -> Path:
    """
    Get the path to the cached embedding npy fileS for an audio file.

    :param model: The name of the model
    :param audio_dir: The path to the audio file
    """
    audio_dir = Path(audio_dir)
    audio_dir = audio_dir.parent / "embeddings" / model
    return list(audio_dir.glob('*.npy'))

def chunk_np_array(np_array, chunk_size, discard_remainder=True):
    """
    Split a NumPy array into chunks of a specified size.

    Parameters:
    - np_array: The input NumPy array to be split.
    - chunk_size: The size of each chunk.
    - discard_remainder: If True, discard any remaining elements that don't fit perfectly into chunks.

    Returns:
    A NumPy array of dimensions (num_chunks, chunk_size)
    """
    if discard_remainder:
        num_chunks = len(np_array) // chunk_size
        output = np.array([np_array[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)])
    else:
        output = np.array([np_array[i:i + chunk_size] for i in range(0, len(np_array), chunk_size)])

    return output