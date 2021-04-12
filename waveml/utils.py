import torch
import numpy as np
import pandas as pd


def to_tensor(X) -> torch.tensor:
    dtype = type(X)

    if dtype == pd.DataFrame:
        return torch.tensor(X.to_numpy())

    elif dtype == pd.Series:
        return torch.tensor(X.values)

    elif dtype == np.ndarray:
        return torch.tensor(X)

    elif dtype == list:
        return torch.tensor(X)

    return X


def to_array(X) -> np.ndarray:
    dtype = type(X)

    if dtype == pd.DataFrame:
        return X.to_numpy()

    elif dtype == pd.Series:
        return X.to_numpy()

    elif dtype == torch.Tensor:
        return X.detach().numpy()

    elif dtype == list:
        return np.array(X)

    return X
