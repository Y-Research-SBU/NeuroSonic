import math
import sys
import os
import time

import torch
import numpy as np

import neurosonic.utils.misc as misc
import neurosonic.utils.lr_sched as lr_sched
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (eeg, audio) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        eeg = eeg.to(device, non_blocking=True).to(torch.float32)
        audio = audio.to(device, non_blocking=True).to(torch.float32)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(audio, eeg)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, data_loader_eval, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    rank = misc.get_rank()
    save_gt = bool(getattr(args, "evaluate_gen", False))
    ema_sampling = int(getattr(args, "ema_sampling", 1))
    if ema_sampling not in (1, 2):
        raise ValueError(f"--ema_sampling must be 1 or 2, got {ema_sampling}")

    # The requested sample count may exceed available eval samples.
    # Also, in distributed mode, each rank should only generate a shard.
    global_target = min(int(args.num_samples), len(data_loader_eval.dataset))
    per_rank_target = global_target if world_size == 1 else math.ceil(global_target / world_size)
    num_steps = per_rank_target // data_loader_eval.batch_size + 1

    # Construct a unique folder per online-eval call to avoid overwriting prior results.
    # Layout:
    #   output_dir/online_eval/<run_id>/<method...-epXXXX[-vYY]>/
    run_id = getattr(args, "run_id", None) or "run"
    base_dir = os.path.join(args.output_dir, "online_eval", str(run_id))
    folder_stem = "{}-steps{}-cfg{}-interval{}-{}-audio{}-tstart{}-ema{}-ep{:04d}".format(
        model_without_ddp.method,
        model_without_ddp.steps,
        model_without_ddp.cfg_scale,
        model_without_ddp.cfg_interval[0],
        model_without_ddp.cfg_interval[1],
        global_target,
        args.eval_t_start,
        ema_sampling,
        int(epoch),
    )

    save_folder = os.path.join(base_dir, folder_stem)
    if os.path.exists(save_folder):
        v = 1
        while True:
            cand = os.path.join(base_dir, f"{folder_stem}-v{v:02d}")
            if not os.path.exists(cand):
                save_folder = cand
                break
            v += 1
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_params = model_without_ddp.ema_params1 if ema_sampling == 1 else model_without_ddp.ema_params2
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = ema_params[i]
    print(f"Switch to ema{ema_sampling}")
    model_without_ddp.load_state_dict(ema_state_dict)

    device = next(model_without_ddp.parameters()).device
    saved = 0
    sum_mse = 0.0
    sum_l1 = 0.0
    count = 0
    time_start = time.perf_counter()
    for step, (eeg, audio_gt) in enumerate(data_loader_eval):
        if saved >= per_rank_target:
            break
        print("Generation step {}/{}".format(step, num_steps))
        eeg = eeg.to(device, non_blocking=True).to(torch.float32)
        audio_gt = audio_gt.to(device, non_blocking=True).to(torch.float32)

        # Evaluation mode: start from GT audio with linear-mix noising at t_start (e.g., 0.8)
        t_start = float(args.eval_t_start)
        eps = torch.randn_like(audio_gt)
        eps_scaled = model_without_ddp.noise_scale * eps
        # z_t = t * x + (1 - t) * eps
        x_init = t_start * audio_gt + (1.0 - t_start) * eps_scaled

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_audio = model_without_ddp.generate(eeg, x_init=x_init, t_start=t_start)

        if misc.get_world_size() > 1:
            torch.distributed.barrier()
        # compute reconstruction metrics
        mse = torch.mean((sampled_audio - audio_gt) ** 2, dim=1)
        l1 = torch.mean(torch.abs(sampled_audio - audio_gt), dim=1)
        sum_mse += mse.sum().item()
        sum_l1 += l1.sum().item()
        count += mse.numel()

        sampled_audio = sampled_audio.detach().float().cpu().numpy()
        audio_gt_np = audio_gt.detach().float().cpu().numpy() if save_gt else None

        for b_id in range(sampled_audio.shape[0]):
            if saved >= per_rank_target:
                break
            # Avoid filename collisions in distributed evaluation.
            base = f"rank{rank:02d}_{saved:06d}"
            if save_gt:
                # Save paired files with same numeric id and letter suffix:
                #   *_g.npy = generated output
                #   *_t.npy = ground truth target
                out_path_g = os.path.join(save_folder, f"{base}_g.npy")
                out_path_t = os.path.join(save_folder, f"{base}_t.npy")
                np.save(out_path_g, sampled_audio[b_id])
                np.save(out_path_t, audio_gt_np[b_id])
            else:
                out_path = os.path.join(save_folder, f"{base}.npy")
                np.save(out_path, sampled_audio[b_id])
            saved += 1

    time_end = time.perf_counter()
    total_sec = time_end - time_start
    if misc.get_world_size() > 1:
        torch.distributed.barrier()
    if misc.get_rank() == 0:
        print("Generation total time: {:.2f} s ({:.3f} s/sample, {} samples)".format(
            total_sec, total_sec / max(1, global_target), global_target))

    # reduce across workers
    if misc.get_world_size() > 1:
        sum_mse = misc.all_reduce_mean(sum_mse) * misc.get_world_size()
        sum_l1 = misc.all_reduce_mean(sum_l1) * misc.get_world_size()
        count = misc.all_reduce_mean(count) * misc.get_world_size()

    if count > 0:
        mse_avg = sum_mse / count
        l1_avg = sum_l1 / count
        print("Eval audio MSE: {:.6f}, L1: {:.6f}".format(mse_avg, l1_avg))
        if log_writer is not None:
            log_writer.add_scalar('eval_audio_mse', mse_avg, epoch)
            log_writer.add_scalar('eval_audio_l1', l1_avg, epoch)

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    if misc.get_world_size() > 1:
        torch.distributed.barrier()
