from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F


PDE_ALIASES = {
    "Burgers": "burger",
    "Darcy": "darcy",
    "Poisson": "poisson",
    "Helmholtz": "helmholtz",
    "NS-NonBounded": "nsnonbounded",
    "Shallow-water": "shallow_water",
    "Heat": "heat",
    "Wave": "wave",
    "Advection-Diffusion": "advection_diffusion",
    "Steady-Heat-Conduction": "steady_heat_conduction",
    "Reaction-Diffusion": "reaction_diffusion",
}


def relative_l2(pred, target, eps=1e-12):
    return float((torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target).clamp_min(eps)).detach().cpu())


def mse(pred, target):
    return float((pred - target).pow(2).mean().detach().cpu())


def mae(pred, target):
    return float((pred - target).abs().mean().detach().cpu())


def obs_relative_l2(pred, target, mask, eps=1e-12):
    diff = (pred - target) * mask
    denom = torch.linalg.vector_norm(target * mask).clamp_min(eps)
    return float((torch.linalg.vector_norm(diff) / denom).detach().cpu())


def pde_residual_norm(residual, eps=1e-12):
    if residual is None:
        return 0.0
    return float((torch.linalg.vector_norm(residual) / max(residual.numel(), 1)).detach().cpu())


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return parse_simple_yaml(text)


def parse_simple_yaml(text):
    config = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                config[key] = parse_scalar(value)
                current = None
            else:
                config[key] = {}
                current = key
            continue
        if current is None:
            continue
        key, value = line.strip().split(":", 1)
        config[current][key.strip()] = parse_scalar(value.strip())
    return config


def parse_scalar(value):
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def canonical_pde_name(config):
    name = config.get("data", {}).get("name", config.get("pde", ""))
    return PDE_ALIASES.get(name, str(name).lower())


def load_result(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            return pickle.load(handle)
    if path.suffix == ".mat":
        return scipy.io.loadmat(path)
    raise ValueError(f"Unsupported result format: {path}")


def load_ground_truth(config, pde, offset):
    data_cfg = config.get("data", {})
    datapath = Path(data_cfg.get("datapath", data_cfg.get("data_path", "")))
    if not datapath:
        raise ValueError("Config must contain data.datapath or data.data_path")

    if pde == "darcy":
        with h5py.File(datapath, "r") as file:
            coef = file["thresh_a_data"][:, :, offset]
            sol = file["thresh_p_data"][:, :, offset]
        return as_bchw(coef, 1), as_bchw(sol, 1), {}

    if pde == "poisson":
        data = scipy.io.loadmat(datapath)
        return as_bchw(data["f_data"][offset], 1), as_bchw(data["phi_data"][offset], 1), {}

    if pde == "helmholtz":
        data = scipy.io.loadmat(datapath)
        params = {"k": torch.tensor([float(data_cfg.get("pdeparam", 1.0))], dtype=torch.float64)}
        return as_bchw(data["f_data"][offset], 1), as_bchw(data["psi_data"][offset], 1), params

    if pde == "nsnonbounded":
        with h5py.File(datapath, "r") as file:
            coef = file["w0"][offset]
            sol = select_final_time(np.asarray(file["w"][offset]))
        return as_bchw(coef, 1), as_bchw(sol, 1), {}

    if pde == "burger":
        data = scipy.io.loadmat(datapath)
        target = data["output"][offset]
        return None, as_bchw(target, 1), {}

    if pde == "shallow_water":
        coef, sol, params = load_shallow_water_gt(datapath, offset)
        return coef, sol, params

    if pde in {"heat", "wave", "advection_diffusion", "steady_heat_conduction"}:
        return load_pair_h5_gt(datapath, offset, pde, data_cfg)

    if pde == "reaction_diffusion":
        return load_reaction_diffusion_gt(datapath, offset)

    raise ValueError(f"Unsupported PDE for evaluation: {pde}")


def load_pair_h5_gt(path, offset, pde, data_cfg):
    with h5py.File(path, "r") as file:
        coef = np.asarray(file["input_data"][offset])
        sol = np.asarray(file["output_data"][offset])
        params = {}
        for name, aliases, default in pair_param_specs(pde):
            value = read_scalar(file, data_cfg, name, aliases, offset, default)
            if value is not None:
                params[name] = torch.tensor([value], dtype=torch.float64)
    return as_bchw(coef), as_bchw(sol), params


def pair_param_specs(pde):
    if pde == "heat":
        return [("alpha", ("fixed_alpha",), 1.0), ("T", (), None), ("total_time", (), None), ("dt", (), None)]
    if pde == "wave":
        return [("c", ("fixed_c",), 1.0), ("T", (), None), ("total_time", (), None), ("dt", (), None)]
    if pde == "advection_diffusion":
        return [("b_x", (), 0.0), ("b_y", (), 0.0), ("kappa", (), 1.0), ("T", (), None), ("total_time", (), None), ("dt", (), None)]
    if pde == "steady_heat_conduction":
        return [("u_D", (), 298.0)]
    return []


def read_scalar(file, data_cfg, name, aliases, offset, default):
    for key in (name, *aliases):
        if key in data_cfg:
            return float(data_cfg[key])
    for key in (name, *aliases):
        if key in file:
            dataset = file[key]
            values = np.asarray(dataset[()] if dataset.shape == () else dataset[:], dtype=np.float64)
            if values.ndim == 0:
                return float(values)
            if values.shape[0] > offset:
                return float(values[offset].reshape(-1)[0])
            if values.shape[0] == 1:
                return float(values.reshape(-1)[0])
        if key in file.attrs:
            return float(np.asarray(file.attrs[key], dtype=np.float64).reshape(-1)[0])
    return default


def load_shallow_water_gt(path, offset):
    pairs = []
    with h5py.File(path, "r") as file:
        if "data" in file and isinstance(file["data"], h5py.Dataset):
            pair = shallow_water_pair_from_dataset(file["data"], offset)
            params = attr_params(file, None, ("g", "eps", "T", "total_time", "dt"))
            channels = pair.shape[0] // 2
            return as_bchw(pair[:channels]), as_bchw(pair[channels:]), params
        for key in sorted(file.keys()):
            if not isinstance(file[key], h5py.Group) or "data" not in file[key]:
                continue
            sample = file[key]
            group = sample["data"]
            params = attr_params(file, sample, ("g", "eps", "T", "total_time", "dt"))
            pair = shallow_water_pair_from_group(group)
            channels = pair.shape[0] // 2
            pairs.append((pair[:channels], pair[channels:], params))
    coef, sol, params = pairs[offset]
    return as_bchw(coef), as_bchw(sol), params


def shallow_water_pair_from_group(group):
    h0 = np.expand_dims(group["h"][0, :, :, 0], axis=0)
    h = np.expand_dims(group["h"][-1, :, :, 0], axis=0)
    if "hu" in group and "hv" in group:
        hu0 = np.expand_dims(group["hu"][0, :, :, 0], axis=0)
        hu = np.expand_dims(group["hu"][-1, :, :, 0], axis=0)
        hv0 = np.expand_dims(group["hv"][0, :, :, 0], axis=0)
        hv = np.expand_dims(group["hv"][-1, :, :, 0], axis=0)
        return np.concatenate([h0, hu0, hv0, h, hu, hv], axis=0)
    return np.concatenate([h0, h], axis=0)


def shallow_water_pair_from_dataset(dataset, offset=0):
    if shallow_water_single_sample_shape(dataset.shape):
        return shallow_water_pair_from_array(dataset[()], 0)
    return shallow_water_pair_from_array(dataset[offset], 0)


def shallow_water_pair_from_array(arr, offset=0):
    arr = np.asarray(arr)
    if arr.ndim == 5:
        sample = arr[offset]
    elif arr.ndim == 4:
        if arr.shape[-1] in (1, 3) or (arr.shape[1] in (1, 3) and arr.shape[-1] == arr.shape[-2]):
            sample = arr
        else:
            sample = arr[offset]
    elif arr.ndim == 3:
        sample = arr
    else:
        raise ValueError(f"Cannot infer shallow_water sample from shape={arr.shape}")
    return np.concatenate([shallow_water_frame(sample, 0), shallow_water_frame(sample, -1)], axis=0)


def shallow_water_single_sample_shape(shape):
    if len(shape) == 3:
        return True
    if len(shape) == 4:
        return shape[-1] in (1, 3) or (shape[1] in (1, 3) and shape[-1] == shape[-2])
    return False


def shallow_water_frame(sample, time_index):
    sample = np.asarray(sample)
    if sample.ndim == 3:
        if sample.shape[0] == sample.shape[1] and sample.shape[2] != sample.shape[1]:
            return np.expand_dims(sample[:, :, time_index], axis=0)
        return np.expand_dims(sample[time_index, :, :], axis=0)
    if sample.ndim == 4:
        if sample.shape[-1] in (1, 3):
            if sample.shape[0] == sample.shape[1] and sample.shape[2] != sample.shape[1]:
                return np.moveaxis(sample[:, :, time_index, :], -1, 0)
            return np.moveaxis(sample[time_index, :, :, :], -1, 0)
        if sample.shape[1] in (1, 3):
            return sample[time_index, :, :, :]
        if sample.shape[0] in (1, 3):
            return sample[:, time_index, :, :]
    raise ValueError(f"Cannot infer shallow_water frame from shape={sample.shape}")


def load_reaction_diffusion_gt(path, offset):
    with h5py.File(path, "r") as file:
        sample_keys = sorted(key for key in file.keys() if isinstance(file[key], h5py.Group) and "data" in file[key])
        if sample_keys:
            sample = file[sample_keys[offset]]
            arr = np.asarray(sample["data"])
            params = reaction_diffusion_params(file, sample)
            init_idx = 50 if Path(path).name == "2D_diff-react_NA_NA.h5" and arr.shape[0] > 50 else 0
            coef = np.stack([arr[init_idx, :, :, 0], arr[init_idx, :, :, 1]], axis=0)
            sol = np.stack([arr[-1, :, :, 0], arr[-1, :, :, 1]], axis=0)
        elif "u" in file and "v" in file:
            u_data = np.asarray(file["u"])
            v_data = np.asarray(file["v"])
            params = reaction_diffusion_params(file, None)
            coef = np.stack([reaction_diffusion_frame(u_data, offset, 0), reaction_diffusion_frame(v_data, offset, 0)], axis=0)
            sol = np.stack([reaction_diffusion_frame(u_data, offset, -1), reaction_diffusion_frame(v_data, offset, -1)], axis=0)
        else:
            raise KeyError(f"{path} must contain sample groups with data or root u/v datasets")
    return as_bchw(coef), as_bchw(sol), params


def reaction_diffusion_frame(arr, offset, time_index):
    sample = np.asarray(arr[offset])
    if sample.ndim == 2:
        return sample
    if sample.ndim == 3:
        if sample.shape[0] == sample.shape[1]:
            return sample[:, :, time_index]
        if sample.shape[1] == sample.shape[2]:
            return sample[time_index, :, :]
    raise ValueError(f"Cannot infer reaction_diffusion frame from shape={sample.shape}")


def reaction_diffusion_params(file, sample):
    params = attr_params(
        file,
        sample,
        (
            "T",
            "total_time",
            "D_u",
            "Du",
            "D_v",
            "Dv",
            "k",
            "x_left",
            "x_right",
            "y_bottom",
            "y_top",
            "dx",
            "dy",
        ),
    )
    for range_name, left_name, right_name in (
        ("x_range", "x_left", "x_right"),
        ("y_range", "y_bottom", "y_top"),
    ):
        if (sample is None or range_name not in sample.attrs) and range_name not in file.attrs:
            continue
        values = attr_value(file, sample, range_name)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size >= 2:
            params.setdefault(left_name, torch.tensor([float(values[0])], dtype=torch.float64))
            params.setdefault(right_name, torch.tensor([float(values[1])], dtype=torch.float64))
    return params


def attr_params(file, sample, names):
    params = {}
    for name in names:
        value = attr_value(file, sample, name)
        if value is not None:
            params[name] = torch.tensor([float(np.asarray(value, dtype=np.float64).reshape(-1)[0])], dtype=torch.float64)
    return params


def attr_value(file, sample, name):
    if sample is not None and name in sample.attrs:
        return sample.attrs[name]
    if name in file.attrs:
        return file.attrs[name]
    return None


def select_final_time(arr):
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, -1]
    raise ValueError(f"Cannot select final time from shape={arr.shape}")


def result_states(result, pde, expected_coef=None, expected_sol=None):
    if "coef_final" in result and "sol_final" in result:
        coef = as_bchw(result["coef_final"], expected_coef)
        sol = as_bchw(result["sol_final"], expected_sol)
        return coef, sol
    if "x_final" in result:
        return None, as_bchw(result["x_final"], expected_sol or 1)
    if "a" in result and "u" in result:
        return as_bchw(result["a"], expected_coef), as_bchw(result["u"], expected_sol)
    raise KeyError("Result must contain coef_final/sol_final, x_final, or a/u")


def as_bchw(value, expected_channels=None):
    tensor = torch.as_tensor(as_numpy(value), dtype=torch.float64)
    while tensor.ndim > 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if expected_channels is not None and tensor.shape[0] == expected_channels:
            tensor = tensor.unsqueeze(0)
        elif expected_channels is None and tensor.shape[-1] == tensor.shape[-2] and tensor.shape[0] <= 16:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4:
        pass
    else:
        raise ValueError(f"Cannot convert shape {tuple(tensor.shape)} to BCHW")
    if expected_channels is not None and tensor.shape[1] != expected_channels:
        if tensor.shape[0] == 1 and tensor.shape[1] > expected_channels:
            tensor = tensor[:, :expected_channels]
        else:
            raise ValueError(f"Expected {expected_channels} channels, got shape {tuple(tensor.shape)}")
    return tensor


def align_channels(pred, target):
    if pred is None or target is None:
        return pred, target
    channels = min(pred.shape[1], target.shape[1])
    return pred[:, :channels], target[:, :channels]


def masks_from_result(result, coef, sol):
    obs = result.get("obs_index", {}) if isinstance(result, dict) else {}
    coef_mask = mask_from_value(obs.get("known_index_a"), coef)
    sol_mask = mask_from_value(obs.get("known_index_u", obs.get("known_sensor")), sol)
    return coef_mask, sol_mask


def mask_from_value(value, reference):
    if reference is None:
        return None
    if value is None:
        return torch.ones_like(reference[:, :1])
    mask = torch.as_tensor(as_numpy(value), dtype=reference.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[0] == 1 and reference.shape[0] > 1:
        mask = mask.repeat(reference.shape[0], 1, 1, 1)
    return mask


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_metrics(config_path, result_path, output_dir=None, offset=None, problem=None):
    config = load_config(config_path)
    pde = canonical_pde_name(config)
    if offset is None:
        offset = infer_offset(result_path, config.get("data", {}).get("offset", 0))
    result = load_result(result_path)
    gt_coef, gt_sol, params = load_ground_truth(config, pde, int(offset))
    pred_coef, pred_sol = result_states(
        result,
        pde,
        expected_coef=None if gt_coef is None else gt_coef.shape[1],
        expected_sol=None if gt_sol is None else gt_sol.shape[1],
    )
    pred_coef, gt_coef = align_channels(pred_coef, gt_coef)
    pred_sol, gt_sol = align_channels(pred_sol, gt_sol)
    coef_mask, sol_mask = masks_from_result(result, pred_coef, pred_sol)
    residual = compute_pde_residual(pde, pred_coef, pred_sol, params, config)

    row = {
        "pde": pde,
        "problem": problem or infer_problem(result_path),
        "offset": int(offset),
        "config_path": str(config_path),
        "result_path": str(result_path),
        "wall_clock_time": float(result.get("time", 0.0)) if isinstance(result, dict) else 0.0,
        "pde_residual_norm": pde_residual_norm(residual),
        "pde_residual_available": residual is not None,
    }
    if pred_coef is not None and gt_coef is not None:
        row.update(metric_block("a", pred_coef, gt_coef, coef_mask))
    if pred_sol is not None and gt_sol is not None:
        row.update(metric_block("u", pred_sol, gt_sol, sol_mask))
    add_loss_curve_tail(row, result)
    write_metrics(row, result_path, output_dir)
    return row


def metric_block(name, pred, target, mask):
    row = {
        f"rel_l2_{name}": relative_l2(pred, target),
        f"mse_{name}": mse(pred, target),
        f"rmse_{name}": math.sqrt(mse(pred, target)),
        f"mae_{name}": mae(pred, target),
    }
    if mask is not None:
        row[f"obs_rel_l2_{name}"] = obs_relative_l2(pred, target, mask)
    return row


def add_loss_curve_tail(row, result):
    if not isinstance(result, dict) or "loss" not in result:
        return
    loss = result["loss"]
    if isinstance(loss, dict):
        for key, values in loss.items():
            if isinstance(values, (list, tuple)) and values:
                row[f"loss_last_{key}"] = float(values[-1])
    elif isinstance(loss, (list, tuple)) and loss:
        row["loss_last"] = float(loss[-1])


def compute_pde_residual(pde, coef, sol, params, config):
    if sol is None:
        return None
    if pde == "poisson":
        return zero_boundary(laplacian(sol) - coef)
    if pde == "helmholtz":
        k = param_value(config.get("data", {}), "pdeparam", 1.0)
        return zero_boundary(laplacian(sol) + (k**2) * sol - coef)
    if pde == "darcy":
        return darcy_residual(coef, sol)
    if pde == "burger":
        return burger_residual(sol)
    if pde == "nsnonbounded":
        return nsnonbounded_residual(coef, sol)
    if pde == "reaction_diffusion":
        return reaction_diffusion_residual(coef, sol, params)
    if pde == "shallow_water":
        if coef.shape[1] != 3 or sol.shape[1] != 3:
            return None
        return shallow_water_residual(coef, sol, params)
    if pde == "heat":
        alpha = param_field(params, "alpha", sol, 1.0)
        time_scale = time_scale_field(params, sol)
        return zero_boundary((sol - coef) / time_scale - alpha * laplacian(0.5 * (coef + sol)))
    if pde == "wave":
        c = param_field(params, "c", sol[:, 0:1], 1.0)
        time_scale = time_scale_field(params, sol[:, 0:1])
        u0, v0 = coef[:, 0:1], coef[:, 1:2]
        u_t, v_t = sol[:, 0:1], sol[:, 1:2]
        res_u = (u_t - u0) / time_scale - 0.5 * (v0 + v_t)
        res_v = (v_t - v0) / time_scale - (c**2) * laplacian(0.5 * (u0 + u_t))
        return torch.cat([zero_boundary(res_u), zero_boundary(res_v)], dim=1)
    if pde == "advection_diffusion":
        bx = param_field(params, "b_x", sol, 0.0)
        by = param_field(params, "b_y", sol, 0.0)
        kappa = param_field(params, "kappa", sol, 1.0)
        time_scale = time_scale_field(params, sol)
        mid = 0.5 * (coef + sol)
        return zero_boundary((sol - coef) / time_scale + bx * dx(mid) + by * dy(mid) - kappa * laplacian(mid))
    if pde == "steady_heat_conduction":
        conductivity = (1.0 + 0.05 * (sol - 298.0)).clamp_min(0.1)
        u_d = param_field(params, "u_D", sol, 298.0)
        return steady_heat_residual_with_boundary(sol, conductivity, coef[:, :1], u_d)
    return None


def param_value(mapping, name, default):
    try:
        return float(mapping.get(name, default))
    except Exception:
        return float(default)


def param_field(params, name, reference, default):
    if name not in params:
        return torch.full((reference.shape[0], 1, 1, 1), float(default), dtype=reference.dtype)
    value = torch.as_tensor(params[name], dtype=reference.dtype).reshape(-1)
    if value.numel() == 1:
        value = value.repeat(reference.shape[0])
    return value.view(reference.shape[0], 1, 1, 1)


def param_field_any(params, names, reference, default):
    for name in names:
        if name in params:
            return param_field(params, name, reference, default)
    return param_field({}, names[0], reference, default)


def float_param(params, name, default):
    if name not in params:
        return float(default)
    value = torch.as_tensor(params[name]).reshape(-1)
    return float(value[0].detach().cpu())


def time_scale_field(params, reference, default=1.0):
    for name in ("T", "total_time", "dt"):
        if name in params:
            return param_field(params, name, reference, 1.0)
    return torch.full((reference.shape[0], 1, 1, 1), float(default), dtype=reference.dtype)


def laplacian(u):
    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    padded = F.pad(u, (1, 1, 1, 1), "constant", 0)
    return (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4.0 * u
    ) / (h**2)


def zero_boundary(x):
    y = x.clone()
    if y.shape[-1] > 1 and y.shape[-2] > 1:
        y[..., 0, :] = 0
        y[..., -1, :] = 0
        y[..., :, 0] = 0
        y[..., :, -1] = 0
    return y


def dx(f):
    h = 1.0 / max(int(f.shape[-1]) - 1, 1)
    padded = F.pad(f, (1, 1, 0, 0), mode="replicate")
    return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * h)


def dy(f):
    h = 1.0 / max(int(f.shape[-2]) - 1, 1)
    padded = F.pad(f, (0, 0, 1, 1), mode="replicate")
    return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * h)


def darcy_residual(a, u):
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=u.dtype).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=u.dtype).view(1, 1, 3, 1) / 2
    grad_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_y = F.conv2d(u, deriv_y, padding=(1, 0))
    return F.conv2d(a * grad_x, deriv_x, padding=(0, 1)) + F.conv2d(a * grad_y, deriv_y, padding=(1, 0)) + 1


def nsnonbounded_residual(a, u):
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=u.dtype).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=u.dtype).view(1, 1, 3, 1) / 2
    residual = F.conv2d(u, deriv_x, padding=(0, 1)) + F.conv2d(u, deriv_y, padding=(1, 0))
    return zero_boundary(residual)


def burger_residual(u):
    deriv_t = torch.tensor([[-1], [0], [1]], dtype=u.dtype).view(1, 1, 3, 1) / 2
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=u.dtype).view(1, 1, 1, 3) / 2
    u_t = F.conv2d(u, deriv_t, padding=(1, 0))
    u_x = F.conv2d(u, deriv_x, padding=(0, 1))
    u_xx = F.conv2d(u_x, deriv_x, padding=(0, 1))
    return u_t + u * u_x - 0.01 * u_xx


def reaction_diffusion_residual(a, u, params):
    if a.shape[1] != 2 or u.shape[1] != 2:
        raise ValueError(f"reaction_diffusion expects 2+2 channels, got a={tuple(a.shape)}, u={tuple(u.shape)}")
    a_u, a_v = a[:, 0:1], a[:, 1:2]
    u_u, u_v = u[:, 0:1], u[:, 1:2]
    time_scale = time_scale_field(params, u_u)
    d_u = param_field_any(params, ("D_u", "Du"), u_u, 2e-3)
    d_v = param_field_any(params, ("D_v", "Dv"), u_v, 4e-3)
    k = param_field(params, "k", u_u, 3e-3)
    res_u = (u_u - a_u) / time_scale - (d_u * neumann_laplacian(u_u, params) + u_u - u_u**3 - k - u_v)
    res_v = (u_v - a_v) / time_scale - (d_v * neumann_laplacian(u_v, params) + u_u - u_v)
    return torch.cat([res_u, res_v], dim=1)


def shallow_water_residual(a, u, params):
    if a.shape[1] != 3 or u.shape[1] != 3:
        raise ValueError(f"shallow_water expects 3+3 channels, got a={tuple(a.shape)}, u={tuple(u.shape)}")
    h0, hu0, hv0 = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    h, hu, hv = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    h_mid = 0.5 * (h0 + h)
    hu_mid = 0.5 * (hu0 + hu)
    hv_mid = 0.5 * (hv0 + hv)
    eps = float_param(params, "eps", 1e-6)
    g = param_field(params, "g", h, 1.0)
    time_scale = time_scale_field(params, h)
    h_safe = h_mid.clamp_min(eps)
    mass = (h - h0) / time_scale + dx(hu_mid) + dy(hv_mid)
    mom_x = (hu - hu0) / time_scale + dx((hu_mid**2) / h_safe + 0.5 * g * h_mid**2) + dy(hu_mid * hv_mid / h_safe)
    mom_y = (hv - hv0) / time_scale + dx(hu_mid * hv_mid / h_safe) + dy((hv_mid**2) / h_safe + 0.5 * g * h_mid**2)
    return torch.cat([mass, mom_x, mom_y], dim=1)


def neumann_laplacian(u, params):
    hx, hy = reaction_diffusion_grid_spacing(params, u)
    padded = F.pad(u, (1, 1, 1, 1), mode="replicate")
    lap_y = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] - 2.0 * u) / (hy**2)
    lap_x = (padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:] - 2.0 * u) / (hx**2)
    return lap_x + lap_y


def reaction_diffusion_grid_spacing(params, reference):
    hx = param_field(params, "dx", reference, 0.0) if "dx" in params else None
    hy = param_field(params, "dy", reference, 0.0) if "dy" in params else None
    x_left, x_right = axis_bounds(params, "x", reference)
    y_bottom, y_top = axis_bounds(params, "y", reference)
    if hx is None:
        hx = (x_right - x_left) / max(int(reference.shape[-1]), 1)
    if hy is None:
        hy = (y_top - y_bottom) / max(int(reference.shape[-2]), 1)
    return hx.abs().clamp_min(1e-12), hy.abs().clamp_min(1e-12)


def axis_bounds(params, axis, reference):
    if axis == "x":
        range_name, left_name, right_name = "x_range", "x_left", "x_right"
        default_left, default_right = -1.0, 1.0
    elif axis == "y":
        range_name, left_name, right_name = "y_range", "y_bottom", "y_top"
        default_left, default_right = -1.0, 1.0
    else:
        raise ValueError(f"Unknown axis={axis!r}")
    if left_name in params and right_name in params:
        return (
            param_field(params, left_name, reference, default_left),
            param_field(params, right_name, reference, default_right),
        )
    if range_name in params:
        values = torch.as_tensor(params[range_name], dtype=reference.dtype).reshape(-1, 2)
        if values.shape[0] == 1:
            values = values.repeat(reference.shape[0], 1)
        if values.shape[0] != reference.shape[0]:
            raise ValueError(f"{range_name} must have one row or batch rows, got shape={tuple(values.shape)}")
        return values[:, 0].view(reference.shape[0], 1, 1, 1), values[:, 1].view(reference.shape[0], 1, 1, 1)
    left = torch.full((reference.shape[0], 1, 1, 1), default_left, dtype=reference.dtype)
    right = torch.full((reference.shape[0], 1, 1, 1), default_right, dtype=reference.dtype)
    return left, right


def steady_heat_residual_with_boundary(u, conductivity, source, u_d):
    residual = nonlinear_heat_conduction_residual(u, conductivity, source)
    h_size = int(u.shape[-2])
    w_size = int(u.shape[-1])
    if h_size <= 0 or w_size <= 0:
        return residual
    dx_val = 1.0 / max(w_size - 1, 1)
    dy_val = 1.0 / max(h_size - 1, 1)
    u_d_row = u_d[..., 0, 0].unsqueeze(-1)
    residual[..., 0, :] = u[..., 0, :] - u_d_row
    if h_size > 1:
        residual[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy_val
    if w_size > 1 and h_size > 2:
        residual[..., 1:-1, 0] = (u[..., 1:-1, 0] - u[..., 1:-1, 1]) / dx_val
        residual[..., 1:-1, -1] = (u[..., 1:-1, -1] - u[..., 1:-1, -2]) / dx_val
    return residual


def nonlinear_heat_conduction_residual(u, conductivity, source):
    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    inv_h2 = 1.0 / (h**2)
    residual = torch.zeros_like(u[:, :1])
    if u.shape[-2] <= 2 or u.shape[-1] <= 2:
        return residual
    center_u = u[:, :, 1:-1, 1:-1]
    center_l = conductivity[:, :, 1:-1, 1:-1]
    accum = torch.zeros_like(center_u)
    for neighbor_u, neighbor_l in (
        (u[:, :, :-2, 1:-1], conductivity[:, :, :-2, 1:-1]),
        (u[:, :, 2:, 1:-1], conductivity[:, :, 2:, 1:-1]),
        (u[:, :, 1:-1, :-2], conductivity[:, :, 1:-1, :-2]),
        (u[:, :, 1:-1, 2:], conductivity[:, :, 1:-1, 2:]),
    ):
        face = 0.5 * (center_l + neighbor_l) * inv_h2
        accum = accum + face * (center_u - neighbor_u)
    residual[:, :, 1:-1, 1:-1] = accum - source[:, :, 1:-1, 1:-1]
    return residual


def infer_offset(path, default):
    stem = Path(path).stem
    matches = re.findall(r"(?:^|_)(\d+)(?:_|$)", stem)
    return int(matches[0]) if matches else int(default)


def infer_problem(path):
    parts = set(Path(path).parts)
    for name in ("forward", "inverse", "both"):
        if name in parts:
            return name
    return ""


def write_metrics(row, result_path, output_dir):
    result_path = Path(result_path)
    out_dir = Path(output_dir) if output_dir is not None else result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{result_path.stem}_metrics_final.json"
    csv_path = out_dir / f"{result_path.stem}_metrics_final.csv"
    json_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, [row])


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def find_results(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted([*path.rglob("*.pkl"), *path.rglob("*.mat")])


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate DiffusionPDE sampling results.")
    parser.add_argument("--config", required=True, help="DiffusionPDE YAML config used for sampling.")
    parser.add_argument("--result", required=True, help="Result .pkl/.mat file or directory containing results.")
    parser.add_argument("--output-dir", default=None, help="Directory for metric files. Defaults to each result directory.")
    parser.add_argument("--offset", type=int, default=None, help="Override dataset offset. By default inferred from result filename.")
    parser.add_argument("--problem", default=None, help="Optional problem label: forward, inverse, or both.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = []
    for result_path in find_results(args.result):
        rows.append(compute_metrics(args.config, result_path, args.output_dir, args.offset, args.problem))
    if args.output_dir is not None:
        write_csv(Path(args.output_dir) / "metrics_all.csv", rows)
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
