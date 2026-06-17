import traceback
import warnings
from pathlib import Path
from typing import NamedTuple, Union
import numpy as np
import scipy.linalg
import torch
import shutil
from hypy_utils import write
from hypy_utils.tqdm_utils import tmap, tq
from pathlib import Path

from kadtk.emb_loader import EmbeddingLoader
from kadtk.model_loader import ModelLoader

warnings.filterwarnings("ignore")
PathLike = Union[str, Path]

class FADInfResults(NamedTuple):
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]

def calc_frechet_distance(
    x: torch.Tensor, 
    y: torch.Tensor,
    cache_dirs: tuple[Path, Path],
    device: str, 
    precision=torch.float32, 
) -> torch.Tensor:
    """FAD implementation in PyTorch.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).
        cache_dirs: Directories to cache embedding statistics.
        device: Device to run the calculation on.
        precision: Type setting for matrix calculation precision.

    Returns:
        The FAD between x and y embedding sets.
    """
    # Load x statistics
    if cache_dirs[0] is not None and cache_dirs[0].exists():
        mu_x = torch.tensor(np.load(cache_dirs[0] / 'mu.npy'), dtype=precision, device=device)
        cov_x = torch.tensor(np.load(cache_dirs[0] / 'cov.npy'), dtype=precision, device=device)
    else:
        x = torch.tensor(x, dtype=precision, device=device)
        mu_x = torch.mean(x, axis=0)
        cov_x = torch.cov(x.T)

        if cache_dirs[0] is not None:
            cache_dirs[0].mkdir(parents=True, exist_ok=True)
            np.save(cache_dirs[0] / 'mu.npy', mu_x.cpu().numpy())
            np.save(cache_dirs[0] / 'cov.npy', cov_x.cpu().numpy())

    # Load y statistics
    if cache_dirs[1] is not None and cache_dirs[1].exists():
        mu_y = torch.tensor(np.load(cache_dirs[1] / 'mu.npy'), dtype=precision, device=device)
        cov_y = torch.tensor(np.load(cache_dirs[1] / 'cov.npy'), dtype=precision, device=device)
    else:
        y = torch.tensor(y,dtype=precision, device=device)
        mu_y = torch.mean(y, axis=0)
        cov_y = torch.cov(y.T)

        if cache_dirs[1] is not None:
            cache_dirs[1].mkdir(parents=True, exist_ok=True)
            np.save(cache_dirs[1] / 'mu.npy', mu_y.cpu().numpy())
            np.save(cache_dirs[1] / 'cov.npy', cov_y.cpu().numpy())

    # Calculate mean distance term
    mu_diff = mu_x-mu_y
    diffnorm_sq = mu_diff@mu_diff

    # Calculate trace term
    cov_prod_np = cov_x.cpu().numpy().dot(cov_y.cpu().numpy())
    covmean_sqrtm, _ = scipy.linalg.sqrtm(cov_prod_np, disp=False)
    if np.iscomplexobj(covmean_sqrtm):
        covmean_sqrtm = covmean_sqrtm.real  # Ensure real values
    tr_covmean = torch.tensor(np.trace(covmean_sqrtm), dtype=precision, device=device)
        
    return diffnorm_sq + torch.trace(cov_x) + torch.trace(cov_y) - 2*tr_covmean

def calc_embd_statistics(embd_lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings.
    """
    return np.mean(embd_lst, axis=0), np.cov(embd_lst, rowvar=False)


class FrechetAudioDistance:
    def __init__(self, ml: ModelLoader, device: str, audio_load_worker: int = 8, logger = None, force_stats_calc=False):
        self.ml = ml
        self.device = torch.device(device)
        self.emb_loader = EmbeddingLoader(ml, load_model=False)
        self.audio_load_worker = audio_load_worker
        self.logger = logger        
        self.force_stats_calc = force_stats_calc

        # Disable gradient calculation because we're not training
        torch.autograd.set_grad_enabled(False)

    def get_cache_dir(self, path: PathLike):
        if self.force_stats_calc:
            return None

        # Check cache stats
        path = Path(path)
        cache_dir = path / "fad_stats" / self.ml.name
        if cache_dir.exists(): 
            if self.force_stats_calc:
                self.logger.info(f"Force recalculate FAD statistics for {path}.")
                shutil.rmtree(cache_dir)
            else:
                self.logger.info(f"FAD statistics is already cached for {path}.")
        return cache_dir

    def score(self, baseline: PathLike, eval: PathLike):
        """
        Calculate a single FAD score between a background and an eval set.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval: Eval matrix or directory containing eval audio files
        """
        bg_cache_dir, eval_cache_dir = self.get_cache_dir(baseline), self.get_cache_dir(eval)
        cache_dirs = (bg_cache_dir, eval_cache_dir)

        embd_bg = self.emb_loader.load_embeddings(baseline)
        embd_eval = self.emb_loader.load_embeddings(eval)
        
        embd_bg = torch.tensor(embd_bg)
        embd_eval = torch.tensor(embd_eval)

        return calc_frechet_distance(embd_bg, embd_eval, cache_dirs, self.device)

    def score_inf(self, baseline: PathLike, eval_files: list[Path], steps: int = 25, min_n = 500, raw: bool = False):
        """
        Calculate FAD for different n (number of samples) and compute FAD-inf.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_files: list of eval audio files
        :param steps: number of steps to use
        :param min_n: minimum n to use
        :param raw: return raw results in addition to FAD-inf
        """
        self.logger.info(f"Calculating FAD-inf for {self.ml.name}...")

        # Load background embeddings
        embd_bg = self.emb_loader.load_embeddings(baseline)
        bg_cache_dir = self.get_cache_dir(baseline)
        cache_dirs = (bg_cache_dir, None)
        
        # If all of the embedding files end in .npy, we can load them directly
        if all([f.suffix == '.npy' for f in eval_files]):
            embeds = [np.load(f) for f in eval_files]
            embeds = np.concatenate(embeds, axis=0)
        else:
            embeds = self.emb_loader._load_embeddings(eval_files, concat=True)
        
        # Calculate maximum n and generate ns
        max_n = len(embeds)
        ns = [int(n) for n in np.linspace(min_n, max_n, steps)]
        
        results = []
        for n in tq(ns, desc="Calculating FAD-inf"):
            # Select n feature frames randomly (with replacement)
            indices = np.random.choice(embeds.shape[0], size=n, replace=True)
            embds_eval = embeds[indices]
            
            embd_bg = torch.tensor(embd_bg)
            embds_eval = torch.tensor(embds_eval)
            score = calc_frechet_distance(embd_bg, embds_eval, cache_dirs=cache_dirs, device=self.device)
            score = score.item()

            # Add to results
            results.append([n, score])

        # Compute FAD-inf based on linear regression of 1/n
        ys = np.array(results)
        xs = 1 / np.array(ns)
        slope, intercept = np.polyfit(xs, ys[:, 1], 1)

        # Compute R^2
        r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2)

        # Since intercept is the FAD-inf, we can just return it
        return FADInfResults(score=intercept, slope=slope, r2=r2, points=results)
    
    def score_individual(self, baseline: PathLike, eval_dir: PathLike, csv_name: Union[Path, str]) -> Path:
        """
        Calculate the FAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_dir: Directory containing eval audio files
        :param csv_name: Name of the csv file to write the results to
        :return: Path to the csv file
        """
        if isinstance(csv_name, str):
            csv = Path(csv_name)
            csv = Path('data') / f'result' / self.ml.name / csv_name
            if csv.exists():
                self.logger.info(f"CSV file {csv} already exists, exiting...")
                return csv
        else:
            csv = Path('data') / f'result' / self.ml.name / "fad-indiv.csv"
        
        # Get cache directory for baseline
        bg_cache_dir = self.get_cache_dir(baseline)
        cache_dirs = (bg_cache_dir, None)

        # Load baseline embeddings
        embd_bg = self.emb_loader.load_embeddings(baseline)
        embd_bg = torch.tensor(embd_bg)

        # Define helper function for calculating z score
        def _find_z_helper(f):
            try:
                # Calculate FAD for individual songs
                embd = self.emb_loader.read_embedding_file(f)
                embd = torch.tensor(embd)
                score = calc_frechet_distance(embd_bg, embd, cache_dirs=cache_dirs, device=self.device)
                return score.item()

            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"An error occurred calculating individual FAD using model {self.ml.name} on file {f}")
                self.logger.error(e)

        # Calculate z score for each eval file
        _files = list(Path(eval_dir).glob("*.*"))
        scores = tmap(_find_z_helper, _files, desc=f"Calculating scores", max_workers=self.audio_load_worker)

        # Write the sorted z scores to csv
        pairs = list(zip(_files, scores))
        pairs = [p for p in pairs if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        write(csv, "\n".join([",".join([str(x).replace(',', '_') for x in row]) for row in pairs]))

        return csv

