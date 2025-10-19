"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import logging

import numpy as np
from sklearn.utils import check_random_state

from . import _emissions
from .base import BaseHMM
from .utils import normalize


__all__ = ["CategoricalHMM"]


_log = logging.getLogger(__name__)
COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


class CategoricalHMM(_emissions.BaseCategoricalHMM, BaseHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    emissionprob_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import CategoricalHMM
    >>> CategoricalHMM(n_components=2)  #doctest: +ELLIPSIS
    CategoricalHMM(algorithm='viterbi',...
    """

    def __init__(
        self,
        n_components=1,
        startprob_prior=1.0,
        transmat_prior=1.0,
        *,
        emissionprob_prior=1.0,
        n_features=None,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="ste",
        init_params="ste",
        implementation="log",
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        emissionprob_prior : array, shape (n_components, n_features), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`emissionprob_`.

        n_features: int, optional
            The number of categorical symbols in the HMM.  Will be inferred
            from the data if not set.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'e' for
            emissionprob.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )
        self.emissionprob_prior = emissionprob_prior
        self.n_features = n_features

    def _init(self, X, lengths=None):
        super()._init(X, lengths)

        self.random_state = check_random_state(self.random_state)

        if self._needs_init("e", "emissionprob_"):
            self.emissionprob_ = self.random_state.rand(
                self.n_components, self.n_features
            )
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super()._check()

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        if self.n_features is None:
            self.n_features = self.emissionprob_.shape[1]
        if self.emissionprob_.shape != (self.n_components, self.n_features):
            raise ValueError(
                f"emissionprob_ must have shape({self.n_components}, {self.n_features})"
            )
        self._check_sum_1("emissionprob_")

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if "e" in self.params:
            self.emissionprob_ = np.maximum(
                self.emissionprob_prior - 1 + stats["obs"], 0
            )
            normalize(self.emissionprob_, axis=1)
