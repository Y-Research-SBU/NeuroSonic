import argparse
import datetime
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.backends.cudnn as cudnn
import neurosonic.utils.misc as misc

import copy
from neurosonic.engine import train_one_epoch, evaluate
from neurosonic.datasets.paired_index import EEGAudioPairedDataset
from neurosonic.datasets.eav_input_images import EAVPreparedEEGAudioDataset

from neurosonic.flow_matching import NeuroSonicFlow


_LOG_FILE_HANDLE = None  # keep log file alive on main process


class _Tee:
    """
    Minimal tee stream: write to multiple streams.
    Useful for saving training logs to a txt while keeping console output.
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                # Best-effort: don't crash training because a log write failed.
                pass
        self.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _unique_ints(xs: Iterable[int]) -> List[int]:
    out = sorted({int(x) for x in xs})
    return out


def _split_subjects(
    subject_ids: Sequence[int],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Cross-subject split (by subject id), deterministic by seed.
    - Shuffle subject_ids, then take last slices as val/test.
    - Clamp counts so val/test are non-empty when ratio>0 and possible.
    """
    subs = _unique_ints(subject_ids)
    n = len(subs)
    if n == 0:
        raise ValueError("No subjects found for splitting.")
    if not (0.0 <= float(val_ratio) < 1.0) or not (0.0 <= float(test_ratio) < 1.0):
        raise ValueError(f"val_ratio/test_ratio must be in [0,1), got {val_ratio}/{test_ratio}")
    if float(val_ratio) + float(test_ratio) >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio}+{test_ratio}")

    rng = np.random.RandomState(int(seed))
    perm = [subs[i] for i in rng.permutation(n)]

    n_test = int(round(n * float(test_ratio)))
    n_val = int(round(n * float(val_ratio)))

    # Ensure at least 1 subject in each requested split when possible.
    if test_ratio > 0 and n >= 2:
        n_test = max(1, n_test)
    if val_ratio > 0 and n >= 2:
        n_val = max(1, n_val)

    # Cap so we always have at least 1 train subject.
    if n_val + n_test >= n:
        overflow = (n_val + n_test) - (n - 1)
        # Prefer shrinking test first, then val.
        shrink_test = min(overflow, n_test)
        n_test -= shrink_test
        overflow -= shrink_test
        n_val = max(0, n_val - overflow)

    # Split by contiguous ranges to avoid Python slicing pitfalls when n_test == 0.
    i_test_start = n - n_test
    i_val_start = n - n_test - n_val
    train_subs = perm[:i_val_start]
    val_subs = perm[i_val_start:i_test_start]
    test_subs = perm[i_test_start:]

    train_subs = sorted(train_subs)
    val_subs = sorted(val_subs)
    test_subs = sorted(test_subs)
    return train_subs, val_subs, test_subs


def _indices_for_subjects_paired_index(dataset, subject_ids: Sequence[int]) -> List[int]:
    subj = np.asarray(dataset.subject_id)
    keep = np.isin(subj, np.asarray(subject_ids, dtype=subj.dtype))
    idx = np.nonzero(keep)[0].astype(np.int64).tolist()
    return idx


def _indices_for_subjects_eav_prepared(dataset, subject_ids: Sequence[int]) -> List[int]:
    # EAVPreparedEEGAudioDataset stores per-subject contiguous blocks in its global index space.
    subs_all = list(getattr(dataset, "_subjects"))
    cum = list(getattr(dataset, "_cum"))
    if len(subs_all) != len(cum):
        raise RuntimeError("Bad EAV prepared dataset internal state (_subjects/_cum).")
    starts = [0] + cum[:-1]
    ranges = {int(sid): (int(st), int(en)) for sid, st, en in zip(subs_all, starts, cum)}

    out: List[int] = []
    for sid in subject_ids:
        st_en = ranges.get(int(sid))
        if st_en is None:
            continue
        st, en = st_en
        out.extend(range(st, en))
    return out


def _create_summary_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:
        print(f"[warn] TensorBoard is unavailable; continuing without SummaryWriter: {e}")
        return None
    return SummaryWriter(log_dir=log_dir)


def get_args_parser():
    parser = argparse.ArgumentParser('NeuroSonic')

    # architecture
    parser.add_argument('--model', default='NeuroSonic-L', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    # audio: 1D array patchify (default)
    parser.add_argument('--audio_len', default=80000, type=int,
                        help='Length of 1D audio feature array')
    parser.add_argument('--audio_patch_len', default=256, type=int,
                        help='Patch length when tokenizing 1D audio (audio_len/audio_patch_len tokens)')
    parser.add_argument('--audio_ext', default='npy', type=str,
                        help='Audio file extension in audio_dir (e.g., npy or wav)')
    parser.add_argument('--audio_sr', default=16000, type=int,
                        help='Expected sampling rate when audio_ext=wav')
    parser.add_argument('--audio_norm', default='peak', type=str, choices=['auto', 'none', 'peak', 'rms'],
                        help='Per-clip gain normalization for audio after loading. auto=wav->peak, npy->none')
    parser.add_argument('--audio_norm_target', default=0.95, type=float,
                        help='Target peak (if audio_norm=peak) or target RMS (if audio_norm=rms)')
    parser.add_argument('--audio_norm_eps', default=1e-8, type=float,
                        help='Epsilon to avoid division by zero in audio normalization')
    parser.add_argument('--audio_norm_max_gain', default=20.0, type=float,
                        help='Max allowed gain for audio normalization (<=0 disables clamp)')
    # legacy 2D mel args (kept for compatibility; unused by current 1D model)
    parser.add_argument('--audio_freq', default=80, type=int)
    parser.add_argument('--audio_time', default=400, type=int)
    parser.add_argument('--audio_patch_f', default=16, type=int)
    parser.add_argument('--audio_patch_t', default=8, type=int)
    parser.add_argument('--eeg_channels', default=30, type=int)
    parser.add_argument('--eeg_time', default=500, type=int)
    parser.add_argument('--eeg_input_channels', default=30, type=int)
    parser.add_argument('--eeg_frame_len', default=50, type=int)
    parser.add_argument('--eeg_hop', default=50, type=int)
    parser.add_argument('--eeg_norm', action='store_true',
                        help='Enable per-sample EEG z-score normalization (per-channel over time)')
    parser.add_argument('--no_eeg_norm', action='store_false', dest='eeg_norm',
                        help='Disable EEG normalization')
    parser.set_defaults(eeg_norm=True)
    parser.add_argument('--eeg_norm_eps', default=1e-6, type=float,
                        help='Epsilon for EEG normalization std clamp')

    # training
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument(
        '--ema_sampling',
        type=int,
        default=1,
        choices=[1, 2],
        help='Which EMA weights to use for generation/eval (1 or 2)',
    )
    parser.add_argument('--P_mean', default=-2.5, type=float)
    parser.add_argument('--P_std', default=1.0, type=float)
    parser.add_argument('--noise_scale', default=0.3, type=float)
    parser.add_argument('--t_eps', default=0.02, type=float)
    parser.add_argument('--t_dist', default='lognormal', type=str,
                        choices=['lognormal', 'sigmoid', 'uniform'],
                        help='Distribution to sample noise level t during training')
    parser.add_argument('--cond_drop_prob', default=0.0, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE sampling method')
    parser.add_argument('--num_sampling_steps', default=100, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_samples', '--num_images', dest='num_samples', default=50000, type=int,
                        help='Number of audio samples to generate/evaluate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--eval_t_start', type=float, default=0.0,
                        help='When evaluating, start from audio_gt with this noise level t (e.g., 0.8)')

    # dataset
    parser.add_argument(
        '--dataset',
        default='eav_input_images',
        type=str,
        choices=['paired_index', 'eav_input_images'],
        help="Dataset backend. paired_index=original EEG npy + audio folder + npz index; "
             "eav_input_images=EAV prepared per-subject npy from Input_images/*.pkl",
    )
    parser.add_argument('--pair_index', default='', type=str)
    parser.add_argument('--eeg_root', default='', type=str)
    parser.add_argument('--audio_dir', default='', type=str)
    parser.add_argument('--audio_index_width', default=6, type=int)
    parser.add_argument('--mmap', action='store_true')
    parser.add_argument(
        '--auto_split',
        action='store_true',
        help='Use cross-subject train/val/test split during training; --evaluate_gen always uses it.',
    )
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.0, type=float)
    parser.add_argument('--split_seed', default=1, type=int)
    parser.add_argument('--eval_split', default='val', type=str, choices=['val', 'test'])

    # EAV Input_images prepared dataset
    parser.add_argument(
        '--eav_prepared_dir',
        default=os.environ.get("EAV_PREPARED_DIR", ""),
        type=str,
        help="Prepared EAV EEG/Audio npy folder containing sub-XX_eeg.npy and sub-XX_audio.npy files",
    )

    # checkpointing
    parser.add_argument('--output_dir', default=os.environ.get("OUTPUT_DIR", "./outputs"),
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=20,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile on the model (may increase memory usage)')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    # Assign a run id and tee stdout/stderr to a txt log (main process only).
    if not hasattr(args, "run_id") or not getattr(args, "run_id"):
        args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if misc.is_main_process() and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, f"train_{args.run_id}.txt")
        try:
            global _LOG_FILE_HANDLE
            _LOG_FILE_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
            sys.stdout = _Tee(sys.stdout, _LOG_FILE_HANDLE)
            sys.stderr = _Tee(sys.stderr, _LOG_FILE_HANDLE)
            print("Logging to:", log_path)
        except Exception as e:
            print(f"[warn] Failed to open log file {log_path}: {e}")

    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = _create_summary_writer(log_dir=args.output_dir)
    else:
        log_writer = None

    if args.dataset == "paired_index":
        dataset_full = EEGAudioPairedDataset(
            index_npz=args.pair_index,
            root_dir=args.eeg_root,
            audio_dir=args.audio_dir,
            audio_len=args.audio_len,
            audio_ext=args.audio_ext,
            audio_sr=args.audio_sr,
            audio_norm=args.audio_norm,
            audio_norm_target=args.audio_norm_target,
            audio_norm_eps=args.audio_norm_eps,
            audio_norm_max_gain=args.audio_norm_max_gain,
            eeg_norm=args.eeg_norm,
            eeg_norm_eps=args.eeg_norm_eps,
            eeg_input_channels=args.eeg_input_channels,
            eeg_time=args.eeg_time,
            audio_index_width=args.audio_index_width,
            mmap=args.mmap,
        )
    elif args.dataset == "eav_input_images":
        dataset_full = EAVPreparedEEGAudioDataset(
            prepared_dir=args.eav_prepared_dir,
            audio_len=args.audio_len,
            audio_norm=args.audio_norm if args.audio_norm != "auto" else "none",
            audio_norm_target=args.audio_norm_target,
            audio_norm_eps=args.audio_norm_eps,
            audio_norm_max_gain=args.audio_norm_max_gain,
            eeg_input_channels=args.eeg_input_channels,
            eeg_time=args.eeg_time,
            eeg_norm=args.eeg_norm,
            eeg_norm_eps=args.eeg_norm_eps,
            mmap=True,  # prepared npy is designed for mmap-friendly loading
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    print(f"Dataset size: {len(dataset_full)}")

    # Cross-subject split (train subjects vs held-out val/test subjects).
    # Generation/evaluation always uses a held-out subject split to avoid
    # accidentally evaluating on the same full dataset used for training.
    dataset_train = dataset_full
    dataset_eval = dataset_full
    force_cross_subject_eval = bool(args.evaluate_gen)
    if args.auto_split or force_cross_subject_eval:
        if args.dataset == "paired_index":
            subjects = _unique_ints(getattr(dataset_full, "subject_id"))
        elif args.dataset == "eav_input_images":
            subjects = _unique_ints(getattr(dataset_full, "_subjects"))
        else:
            subjects = []

        train_subs, val_subs, test_subs = _split_subjects(
            subjects,
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.split_seed),
        )
        if len(train_subs) == 0:
            raise RuntimeError("Auto-split produced empty train subject list.")
        if args.eval_split == "test" and len(test_subs) == 0:
            raise RuntimeError("--eval_split test requested but test split is empty; set --test_ratio > 0.")
        if args.eval_split == "val" and len(val_subs) == 0:
            raise RuntimeError("--eval_split val requested but val split is empty; set --val_ratio > 0.")

        if args.dataset == "paired_index":
            train_idx = _indices_for_subjects_paired_index(dataset_full, train_subs)
            val_idx = _indices_for_subjects_paired_index(dataset_full, val_subs) if val_subs else []
            test_idx = _indices_for_subjects_paired_index(dataset_full, test_subs) if test_subs else []
        else:
            train_idx = _indices_for_subjects_eav_prepared(dataset_full, train_subs)
            val_idx = _indices_for_subjects_eav_prepared(dataset_full, val_subs) if val_subs else []
            test_idx = _indices_for_subjects_eav_prepared(dataset_full, test_subs) if test_subs else []

        dataset_train = torch.utils.data.Subset(dataset_full, train_idx)
        if args.eval_split == "test":
            dataset_eval = torch.utils.data.Subset(dataset_full, test_idx)
        else:
            dataset_eval = torch.utils.data.Subset(dataset_full, val_idx)

        print(
            "Cross-subject split:",
            f"mode={'forced-eval' if force_cross_subject_eval else 'train'};",
            f"subjects(total/train/val/test)={len(subjects)}/{len(train_subs)}/{len(val_subs)}/{len(test_subs)};",
            f"samples(train/eval)={len(dataset_train)}/{len(dataset_eval)};",
            f"seed={args.split_seed}, val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, eval_split={args.eval_split}",
        )

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create the conditional flow matching model.
    model = NeuroSonicFlow(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)
    if args.compile:
        model = torch.compile(model)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    def _find_latest_checkpoint(resume_dir: str) -> str | None:
        """
        Find the latest checkpoint path in resume_dir.
        Backward compatible order:
          0) if resume_dir is a direct .pth file path, use it
          1) checkpoint-last-epXXXX.pth (new non-overwriting "last" series)
          2) checkpoint-last.pth (legacy fixed name)
          3) checkpoint-<epoch>.pth (periodic epoch snapshots)
        Returns the chosen checkpoint path or None if nothing found.
        """
        if not resume_dir:
            return None
        p = Path(resume_dir)
        if p.exists() and p.is_file() and p.suffix == ".pth":
            return str(p)
        d = p
        if not d.exists() or not d.is_dir():
            return None

        def _pick_max_epoch(glob_pat: str, prefix: str) -> Path | None:
            best_p: Path | None = None
            best_ep: int = -1
            for p in d.glob(glob_pat):
                name = p.name
                if not name.startswith(prefix) or not name.endswith(".pth"):
                    continue
                mid = name[len(prefix):-4]
                # last-ep0005 -> "0005", checkpoint-100 -> "100"
                if mid.isdigit():
                    ep = int(mid)
                    if ep > best_ep:
                        best_ep = ep
                        best_p = p
            return best_p

        p_last_series = _pick_max_epoch("checkpoint-last-ep*.pth", "checkpoint-last-ep")
        if p_last_series is not None:
            return str(p_last_series)

        legacy = d / "checkpoint-last.pth"
        if legacy.exists():
            return str(legacy)

        p_epoch = _pick_max_epoch("checkpoint-*.pth", "checkpoint-")
        if p_epoch is not None:
            return str(p_epoch)

        # Fallback to newest by mtime for any checkpoint-*.pth
        candidates = list(d.glob("checkpoint-*.pth"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(candidates[0])
        return None

    # Resume from checkpoint if provided
    checkpoint_path = _find_latest_checkpoint(args.resume)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint:", checkpoint_path)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # DataLoader for online eval during training (when --online_eval)
    data_loader_eval = None
    if args.online_eval:
        if args.distributed:
            sampler_eval = torch.utils.data.DistributedSampler(
                dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval, sampler=sampler_eval,
            batch_size=args.gen_bsz,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        sampler_eval = torch.utils.data.DistributedSampler(
            dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval, sampler=sampler_eval,
            batch_size=args.gen_bsz,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, data_loader_eval, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                # do not overwrite: keep a unique "last series" file per save point
                epoch_name=f"last-ep{epoch:04d}"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, data_loader_eval, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
