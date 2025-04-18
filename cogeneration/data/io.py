import io
import json
import os
import pickle
from typing import Any

import numpy as np
import torch


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def write_pkl(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, "rb") as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(
                    f"Error. Failed to read {read_path}. First error: {e}\n Second error: {e2}"
                )
            raise (e)


def write_numpy_json(file_path: str, data: Any):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            return super().default(obj)

    with open(file_path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)
