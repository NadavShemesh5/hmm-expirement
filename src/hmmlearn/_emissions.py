import functools
import inspect
import warnings

import numpy as np
from sklearn.utils import check_random_state

from .base import _AbstractHMM


_CATEGORICALHMM_DOC_SUFFIX = """

Notes
-----
Unlike other HMM classes, `CategoricalHMM` ``X`` arrays have shape
``(n_samples, 1)`` (instead of ``(n_samples, n_features)``).  Consider using
`sklearn.preprocessing.LabelEncoder` to transform your input to the right
format.
"""


def _make_wrapper(func):
    return functools.wraps(func)(lambda *args, **kwargs: func(*args, **kwargs))


class BaseCategoricalHMM(_AbstractHMM):
    def __init_subclass__(cls):
        for name in [
            "decode",
            "fit",
            "predict",
            "predict_proba",
            "sample",
            "score",
            "score_samples",
        ]:
            meth = getattr(cls, name)
            doc = inspect.getdoc(meth)
            if doc is None or _CATEGORICALHMM_DOC_SUFFIX in doc:
                wrapper = meth
            else:
                wrapper = _make_wrapper(meth)
                wrapper.__doc__ = (
                    doc.replace("(n_samples, n_features)", "(n_samples, 1)")
                    + _CATEGORICALHMM_DOC_SUFFIX
                )
            setattr(cls, name, wrapper)

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a categorical distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if self.n_features is not None:
            if self.n_features - 1 < X.max():
                raise ValueError(
                    f"Largest symbol is {X.max()} but the model only emits "
                    f"symbols up to {self.n_features - 1}"
                )
        else:
            self.n_features = X.max() + 1

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _compute_likelihood(self, X):
        if X.shape[1] != 1:
            warnings.warn(
                "Inputs of shape other than (n_samples, 1) are deprecated.",
                DeprecationWarning,
            )
            X = np.concatenate(X)[:, None]
        return self.emissionprob_[:, X.squeeze(1)].T

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats["obs"] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats=stats,
            X=X,
            lattice=lattice,
            posteriors=posteriors,
            fwdlattice=fwdlattice,
            bwdlattice=bwdlattice,
        )

        if "e" in self.params:
            if X.shape[1] != 1:
                warnings.warn(
                    "Inputs of shape other than (n_samples, 1) are deprecated.",
                    DeprecationWarning,
                )
                X = np.concatenate(X)[:, None]
            np.add.at(stats["obs"].T, X.squeeze(1), posteriors)

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]
