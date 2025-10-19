"""Private utilities."""

import numpy as np


def split_X_lengths(X, lengths):
    if lengths is None:
        return [X]
    else:
        cs = np.cumsum(lengths)
        n_samples = len(X)
        if cs[-1] != n_samples:
            raise ValueError(
                f"lengths array {lengths} doesn't sum to {n_samples} samples"
            )
        return np.split(X, cs)[:-1]
