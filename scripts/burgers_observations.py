"""Observation layouts for Burgers fields stored as (physical time, space)."""

import hashlib
from pathlib import Path

import torch


def make_burgers_mask(data_config, shape=(128, 128), device="cpu"):
    """Build a mask independently of sampling RNG and batch/resume order.

    Paper layouts use the same per-input CPU RNG as FM4PDEbaseline: sensor
    seed, test split, dataset filename and absolute sample offset identify
    the mask. Configs without sensor_mode retain the original sensor columns.
    """
    mode = data_config.get("sensor_mode", "sensor_columns")
    time_size, space_size = shape
    if mode == "sensor_columns":
        count = data_config.get("num_sensor_columns", 5)
        size = space_size
        generator = torch.Generator(device=device).manual_seed(0)
        mask = torch.zeros(shape, dtype=torch.float64, device=device)
    else:
        if mode == "random":
            count, size = data_config.get("obs_size", 500), time_size * space_size
        elif mode == "time_slices":
            count, size = data_config.get("num_time_slices", 5), time_size
        else:
            raise ValueError(f"Unknown Burgers sensor_mode: {mode!r}")
        seed = int(data_config.get("mask_seed", 1))
        split = str(data_config.get("split", "test")).lower()
        sample_id = f"{Path(data_config['datapath']).name}:{int(data_config['offset'])}"
        payload = f"mask|{seed}|{split}|{sample_id}|0".encode("utf-8")
        mask_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(mask_seed)
        mask = torch.zeros(shape, dtype=torch.float64, device="cpu")

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= size:
        raise ValueError(f"Burgers {mode} count must be an integer in [1, {size}], got {count!r}")
    indices = torch.randperm(size, generator=generator, device=mask.device)[:count]
    if mode == "random":
        mask.view(-1)[indices] = 1
    elif mode == "time_slices":
        mask[indices, :] = 1
    else:
        mask[:, indices] = 1
    return mask.to(device)
