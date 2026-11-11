import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


##########################
# Data Loading Functions #
##########################

def load_json_data(file_path: str) -> dict:
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(file_path)


def save_json_data(
    data: dict,
    file_path: str,
    encoding: str = "utf-8",
    indent: int = 4,
) -> None:
    """Save JSON data, creating parent directories when necessary."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_merge_json_data(file_paths: list) -> dict:
    """Merge JSON dictionaries; later files overwrite duplicate keys."""
    merged_data = {}

    for file_path in file_paths:
        merged_data.update(load_json_data(file_path))

    return merged_data


def load_merge_csv_data(file_paths: list) -> pd.DataFrame:
    """Concatenate CSV files, returning an empty DataFrame for no files."""
    data_frames = [load_csv_data(path) for path in file_paths]

    if not data_frames:
        return pd.DataFrame()

    return pd.concat(data_frames, ignore_index=True)


###############################
# Device Management Functions #
###############################

def _backend_available(device_type: str) -> bool:
    """Check backend availability, including on older PyTorch builds."""
    if device_type == "cpu":
        return True

    if device_type == "mps":
        backend = getattr(torch.backends, "mps", None)
    elif device_type in ("cuda", "xpu"):
        backend = getattr(torch, device_type, None)
    else:
        return False

    return backend is not None and backend.is_available()


def get_device(preferred=None) -> torch.device:
    """
    Select an available device.

    With no preference, preserve the original priority and add XPU:
        MPS -> CUDA -> XPU -> CPU

    Examples:
        get_device()         # Automatic selection
        get_device("xpu")    # Require an Intel GPU
        get_device("xpu:0")  # Require Intel GPU 0
        get_device("cuda:0")
        get_device("cpu")

    An explicitly requested unavailable device raises an error.
    """
    if preferred is None:
        for device_type in ("mps", "cuda", "xpu", "cpu"):
            if _backend_available(device_type):
                return torch.device(device_type)

    device = torch.device(preferred)

    if device.type not in ("cpu", "cuda", "xpu", "mps"):
        raise ValueError(f"Unsupported device type: {device.type}")

    if not _backend_available(device.type):
        raise RuntimeError(
            f"Requested device '{device}' is unavailable. "
            "Check your PyTorch build, GPU drivers, and hardware."
        )

    if device.type in ("cuda", "xpu") and device.index is not None:
        backend = getattr(torch, device.type)
        device_count = backend.device_count()

        if not 0 <= device.index < device_count:
            raise ValueError(
                f"Invalid device index {device.index}: "
                f"{device_count} {device.type} device(s) available."
            )

    return device


def move_to_device(data, device: torch.device, non_blocking=False):
    """
    Move tensors or models to a device, including nested containers.

    Non-tensor values such as strings, numbers, and None are preserved.
    """
    device = torch.device(device)

    if isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)

    if isinstance(data, torch.nn.Module):
        return data.to(device=device, non_blocking=non_blocking)

    if isinstance(data, dict):
        return {
            key: move_to_device(value, device, non_blocking)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            move_to_device(item, device, non_blocking)
            for item in data
        ]

    if isinstance(data, tuple):
        items = [
            move_to_device(item, device, non_blocking)
            for item in data
        ]

        # Preserve named tuples.
        if hasattr(data, "_fields"):
            return type(data)(*items)

        return tuple(items)

    return data


def print_device_info(device: torch.device) -> None:
    """Print information about the selected CPU or GPU."""
    device = get_device(device)
    print(f"Device: {device}")

    if device.type in ("cuda", "xpu"):
        backend = getattr(torch, device.type)
        index = (
            device.index
            if device.index is not None
            else backend.current_device()
        )
        properties = backend.get_device_properties(index)
        gib = 1024 ** 3

        print(f"Backend: {device.type.upper()}")
        print(f"Device index: {index}")
        print(f"GPU name: {backend.get_device_name(index)}")
        print(f"GPU count: {backend.device_count()}")
        print(f"Total memory: {properties.total_memory / gib:.2f} GiB")
        print(
            f"Allocated memory: "
            f"{backend.memory_allocated(index) / gib:.2f} GiB"
        )
        print(
            f"Reserved memory: "
            f"{backend.memory_reserved(index) / gib:.2f} GiB"
        )

    elif device.type == "mps":
        print("MPS backend is active")

        memory_functions = (
            ("Recommended max memory", "recommended_max_memory"),
            ("Current allocated memory", "current_allocated_memory"),
            ("Driver allocated memory", "driver_allocated_memory"),
        )

        for label, function_name in memory_functions:
            function = getattr(torch.mps, function_name, None)

            if callable(function):
                print(f"{label}: {function() / (1024 ** 3):.2f} GiB")

    else:
        print("CPU backend is active")


def maybe_empty_cache(device: torch.device) -> None:
    """
    Release unused GPU allocator cache.

    This does not free memory occupied by live tensors or models.
    """
    device = torch.device(device)

    if not _backend_available(device.type):
        return

    if device.type in ("cuda", "xpu"):
        backend = getattr(torch, device.type)

        # Respect the requested GPU index.
        with backend.device(device):
            backend.empty_cache()

    elif device.type == "mps":
        empty_cache = getattr(torch.mps, "empty_cache", None)

        if callable(empty_cache):
            empty_cache()


#####################
# Seeding Functions #
#####################

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch seeds its CPU and supported accelerator generators.
    torch.manual_seed(seed)