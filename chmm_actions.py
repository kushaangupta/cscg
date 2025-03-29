from __future__ import print_function
from builtins import range
import numpy as np
import numba as nb
from tqdm import trange
import sys


def validate_seq(x, a, n_clones=None):
    """Validate an input sequence of observations x and actions a"""
    assert len(x) == len(a) > 0
    assert len(x.shape) == len(a.shape) == 1, "Flatten your array first"
    assert x.dtype == a.dtype == np.int64
    assert 0 <= x.min(), "Number of emissions inconsistent with training sequence"
    if n_clones is not None:
        assert len(n_clones.shape) == 1, "Flatten your array first"
        assert n_clones.dtype == np.int64
        assert all(
            [c > 0 for c in n_clones]
        ), "You can't provide zero clones for any emission"
        n_emissions = n_clones.shape[0]
        assert (
            x.max() < n_emissions
        ), "Number of emissions inconsistent with training sequence"


def datagen_structured_obs_room(
    room,
    start_r=None,
    start_c=None,
    no_left=[],
    no_right=[],
    no_up=[],
    no_down=[],
    length=10000,
    seed=42,
):
    """room is a 2d numpy array. inaccessible locations are marked by -1.
    start_r, start_c: starting locations

    In addition, there are invisible obstructions in the room
    which disallows certain actions from certain states.

    no_left:
    no_right:
    no_up:
    no_down:

    Each of the above are list of states from which the corresponding action is not allowed.

    """
    np.random.seed(seed)
    H, W = room.shape
    if start_r is None or start_c is None:
        start_r, start_c = np.random.randint(H), np.random.randint(W)

    actions = np.zeros(length, int)
    x = np.zeros(length, int)  # observations
    rc = np.zeros((length, 2), int)  # actual r&c

    r, c = start_r, start_c
    x[0] = room[r, c]
    rc[0] = r, c

    count = 0
    while count < length - 1:

        act_list = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
        if (r, c) in no_left:
            act_list.remove(0)
        if (r, c) in no_right:
            act_list.remove(1)
        if (r, c) in no_up:
            act_list.remove(2)
        if (r, c) in no_down:
            act_list.remove(3)

        a = np.random.choice(act_list)

        # Check for actions taking out of the matrix boundary.
        prev_r = r
        prev_c = c
        if a == 0 and 0 < c:
            c -= 1
        elif a == 1 and c < W - 1:
            c += 1
        elif a == 2 and 0 < r:
            r -= 1
        elif a == 3 and r < H - 1:
            r += 1

        # Check whether action is taking to inaccessible states.
        temp_x = room[r, c]
        if temp_x == -1:
            r = prev_r
            c = prev_c
            pass

        actions[count] = a
        x[count + 1] = room[r, c]
        rc[count + 1] = r, c
        count += 1

    return actions, x, rc


class CHMM(object):
    """
    Conditional Hidden Markov Model with action sequences.

    The model represents a system where transitions between hidden states
    are conditioned on actions, and observations are emitted from hidden states.

    Parameters
    ----------
    n_clones : numpy.ndarray
        Number of hidden states (clones) for each observation type.
    x : numpy.ndarray
        Training observation sequence.
    a : numpy.ndarray
        Training action sequence.
    pseudocount : float, optional
        Pseudocount for smoothing probabilities, defaults to 0.0.
    dtype : numpy.dtype, optional
        Data type for model parameters, defaults to np.float32.
    seed : int, optional
        Random seed for initialization, defaults to 42.
        
    Attributes
    ----------
    n_clones : numpy.ndarray
        Number of hidden states (clones) for each observation type, shape
        (n_observations,).
    pseudocount : float
        Pseudocount for smoothing probabilities.
    dtype : numpy.dtype
        Data type for model parameters.
    C : numpy.ndarray
        Transition count matrix, shape (n_actions, n_states, n_states).
    T : numpy.ndarray
        Transition probability matrix, shape (n_actions, n_states, n_states).
    Pi_x : numpy.ndarray
        Initial state distribution.
    Pi_a : numpy.ndarray
        Action probability distribution.
    """
    def __init__(self, n_clones, x, a, pseudocount=0.0, dtype=np.float32, seed=42):
        """Construct a CHMM objct. n_clones is an array where n_clones[i] is the
        number of clones assigned to observation i. x and a are the observation sequences
        and action sequences, respectively."""
        np.random.seed(seed)
        self.n_clones = n_clones
        validate_seq(x, a, self.n_clones)
        assert pseudocount >= 0.0, "The pseudocount should be positive"
        print("Average number of clones:", n_clones.mean())
        self.pseudocount = pseudocount
        self.dtype = dtype
        n_states = self.n_clones.sum()  # total number of hidden states
        n_actions = a.max() + 1
        self.C = np.random.rand(n_actions, n_states, n_states).astype(dtype)
        self.Pi_x = np.ones(n_states) / n_states  # shape (n_observations,)
        self.Pi_a = np.ones(n_actions) / n_actions  # initial action distribution
        self.update_T()

    def update_T(self):
        """
        Update transition probability matrix from accumulated transition counts.

        Updates self.T based on the current counts in self.C,
        applying pseudocounts and normalization.
        """
        self.T = self.C + self.pseudocount
        norm = self.T.sum(2, keepdims=True)
        norm[norm == 0] = 1
        self.T /= norm

    # def update_T(self):
    #     self.T = self.C + self.pseudocount
    #     norm = self.T.sum(2, keepdims=True)  # old model (conditional on actions)
    #     norm[norm == 0] = 1
    #     self.T /= norm
    #     norm = self.T.sum((0, 2), keepdims=True)  # new model (generates actions too)
    #     norm[norm == 0] = 1
    #     self.T /= norm

    def update_E(self, CE):
        """
        Update emission probability matrix from counts.

        Parameters
        ----------
        CE : numpy.ndarray
            Emission count matrix, shape (n_states, n_emissions).

        Returns
        -------
        E : numpy.ndarray
            Normalized emission probability matrix.
        """
        E = CE + self.pseudocount
        norm = E.sum(1, keepdims=True)
        norm[norm == 0] = 1
        E /= norm
        return E

    def bps(self, x, a):
        """
        Compute bits per symbol for a sequence of observations and actions.
        
        Parameters
        ----------
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.
            
        Returns
        -------
        float
            Negative log2 likelihood (bits per symbol).
        """
        validate_seq(x, a, self.n_clones)
        log2_lik = forward(self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a)[0]
        return -log2_lik

    def bpsE(self, E, x, a):
        """
        Compute bits per symbol using alternate emission matrix.
        
        Parameters
        ----------
        E : numpy.ndarray
            Alternative emission probability matrix.
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.
            
        Returns
        -------
        float
            Negative log2 likelihood (bits per symbol).
        """
        validate_seq(x, a, self.n_clones)
        log2_lik = forwardE(
            self.T.transpose(0, 2, 1), E, self.Pi_x, self.n_clones, x, a
        )
        return -log2_lik

    def bpsV(self, x, a):
        """
        Compute bits per symbol using max-product algorithm.

        Parameters
        ----------
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.

        Returns
        -------
        float
            Negative log2 likelihood (bits per symbol).
        """
        validate_seq(x, a, self.n_clones)
        log2_lik = forward_mp(
            self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a
        )[0]
        return -log2_lik

    def decode(self, x, a):
        """
        Compute most likely state sequence using Viterbi algorithm.
        
        Parameters
        ----------
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.
            
        Returns
        -------
        log2_lik : float
            Negative log2 likelihood of the sequence.
        states : numpy.ndarray
            Most likely state sequence.
        """
        log2_lik, mess_fwd = forward_mp(
            self.T.transpose(0, 2, 1),
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
        return -log2_lik, states

    def decodeE(self, E, x, a):
        """
        Compute most likely state sequence with alternate emission matrix.

        Compute the MAP assignment of latent variables using max-product message
        passing with an alternative emission matrix.

        Parameters
        ----------
        E : numpy.ndarray
            Alternative emission probability matrix.
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.

        Returns
        -------
        log2_lik : float
            Negative log2 likelihood of the sequence.
        states : numpy.ndarray
            Most likely state sequence.
        """
        log2_lik, mess_fwd = forwardE_mp(
            self.T.transpose(0, 2, 1),
            E,
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states = backtraceE(self.T, E, self.n_clones, x, a, mess_fwd)
        return -log2_lik, states

    def learn_em_T(self, x, a, n_iter=100, term_early=True):
        """
        Train transition matrix using Expectation-Maximization algorithm.

        Run EM training, keeping E deterministic and fixed, learning T
        
        Parameters
        ----------
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.
        n_iter : int, optional
            Maximum number of iterations, defaults to 100.
        term_early : bool, optional
            Whether to terminate early if likelihood decreases, defaults to True.
            
        Returns
        -------
        convergence : list
            Bits per symbol at each iteration.
        """
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for it in pbar:
            # E
            log2_lik, mess_fwd = forward(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backward(self.T, self.n_clones, x, a)
            updateC(self.C, self.T, self.n_clones, mess_fwd, mess_bwd, x, a)
            # M
            self.update_T()
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                if term_early:
                    break
            log2_lik_old = log2_lik.mean()
        return convergence

    def learn_viterbi_T(self, x, a, n_iter=100):
        """
        Train transition matrix using Viterbi training.

        Run Viterbi training, keeping E deterministic and fixed, learning T

        Parameters
        ----------
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.
        n_iter : int, optional
            Maximum number of iterations, defaults to 100.

        Returns
        -------
        convergence : list
            Bits per symbol at each iteration.
        """
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for it in pbar:
            # E
            log2_lik, mess_fwd = forward_mp(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
            self.C[:] = 0
            for t in range(1, len(x)):
                aij, i, j = (
                    a[t - 1],
                    states[t - 1],
                    states[t],
                )  # at time t-1 -> t we go from observation i to observation j
                self.C[aij, i, j] += 1.0
            # M
            self.update_T()

            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        return convergence

    def learn_em_E(self, x, a, n_iter=100, pseudocount_extra=1e-20):
        """
        Train emission matrix using Expectation-Maximization algorithm.

        Run Viterbi training, keeping T fixed, learning E

        Parameters
        ----------
        x : numpy.ndarray
            Observation sequence.
        a : numpy.ndarray
            Action sequence.
        n_iter : int, optional
            Maximum number of iterations, defaults to 100.
        pseudocount_extra : float, optional
            Additional pseudocount for emission matrix, defaults to 1e-20.

        Returns
        -------
        convergence : list
            Bits per symbol at each iteration.
        E : numpy.ndarray
            Trained emission probability matrix.
        """
        sys.stdout.flush()
        n_emissions, n_states = len(self.n_clones), self.n_clones.sum()
        CE = np.ones((n_states, n_emissions), self.dtype)
        E = self.update_E(CE + pseudocount_extra)
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for it in pbar:
            # E
            log2_lik, mess_fwd = forwardE(
                self.T.transpose(0, 2, 1),
                E,
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backwardE(self.T, E, self.n_clones, x, a)
            updateCE(CE, E, self.n_clones, mess_fwd, mess_bwd, x, a)
            # M
            E = self.update_E(CE + pseudocount_extra)
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        return convergence, E

    def sample(self, length):
        """
        Generate a random sample from the model.
        
        Parameters
        ----------
        length : int
            Length of the sequence to generate.
            
        Returns
        -------
        sample_x : numpy.ndarray
            Generated observation sequence.
        sample_a : numpy.ndarray
            Generated action sequence.
        """
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)
        sample_x = np.zeros(length, dtype=np.int64)
        sample_a = np.random.choice(len(self.Pi_a), size=length, p=self.Pi_a)

        # Sample
        p_h = self.Pi_x
        for t in range(length):
            h = np.random.choice(len(p_h), p=p_h)
            sample_x[t] = np.digitize(h, state_loc) - 1
            p_h = self.T[sample_a[t], h]
        return sample_x, sample_a

    def sample_sym(self, sym, length):
        """
        Generate a random sample from the model starting with a specific observation.
        
        Parameters
        ----------
        sym : int
            Initial observation symbol.
        length : int
            Length of the sequence to generate.
            
        Returns
        -------
        seq : list
            Generated observation sequence.
        """
        # Prepare structures
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)

        seq = [sym]

        alpha = np.ones(self.n_clones[sym])
        alpha /= alpha.sum()

        for _ in range(length):
            obs_tm1 = seq[-1]
            T_weighted = self.T.sum(0)

            long_alpha = np.dot(
                alpha, T_weighted[state_loc[obs_tm1] : state_loc[obs_tm1 + 1], :]
            )
            long_alpha /= long_alpha.sum()
            idx = np.random.choice(np.arange(self.n_clones.sum()), p=long_alpha)

            sym = np.digitize(idx, state_loc) - 1
            seq.append(sym)

            temp_alpha = long_alpha[state_loc[sym] : state_loc[sym + 1]]
            temp_alpha /= temp_alpha.sum()
            alpha = temp_alpha

        return seq

    def bridge(self, state1, state2, max_steps=100):
        """
        Find optimal action sequence to transition from state1 to state2.
        
        Parameters
        ----------
        state1 : int
            Starting state.
        state2 : int
            Target state.
        max_steps : int, optional
            Maximum number of steps allowed, defaults to 100.
            
        Returns
        -------
        s_a : tuple
            Tuple of (actions, states) representing the optimal path.
        """
        Pi_x = np.zeros(self.n_clones.sum(), dtype=self.dtype)
        Pi_x[state1] = 1
        log2_lik, mess_fwd = forward_mp_all(
            self.T.transpose(0, 2, 1), Pi_x, self.Pi_a, self.n_clones, state2, max_steps
        )
        s_a = backtrace_all(self.T, self.Pi_a, self.n_clones, mess_fwd, state2)
        return s_a


def updateCE(CE, E, n_clones, mess_fwd, mess_bwd, x, a):
    """
    Update emission matrix counts based on forward-backward messages.
    
    Parameters
    ----------
    CE : numpy.ndarray
        Matrix to store emission counts, shape (n_states, n_emissions).
    E : numpy.ndarray
        Current emission probability matrix.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    mess_fwd : numpy.ndarray
        Forward messages from forward algorithm.
    mess_bwd : numpy.ndarray
        Backward messages from backward algorithm.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
        
    Notes
    -----
    Updates CE in-place with emission counts based on forward-backward algorithm.
    """
    timesteps = len(x)
    gamma = mess_fwd * mess_bwd
    norm = gamma.sum(1, keepdims=True)
    norm[norm == 0] = 1
    gamma /= norm
    CE[:] = 0
    for t in range(timesteps):
        CE[:, x[t]] += gamma[t]


def forwardE(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """
    Forward algorithm with explicit emission matrix.

    Log-probability of a sequence, and optionally, messages

    Parameters
    ----------
    T_tr : numpy.ndarray
        Transposed transition matrix, shape (n_actions, n_states, n_states).
    E : numpy.ndarray
        Emission probability matrix, shape (n_states, n_emissions).
    Pi : numpy.ndarray
        Initial state distribution.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
    store_messages : bool, optional
        Whether to store and return forward messages.

    Returns
    -------
    log2_lik : numpy.ndarray
        Log2 probability of each observation.
    mess_fwd : numpy.ndarray, optional
        Forward messages if store_messages is True.
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = T_tr[aij].dot(message)
        message *= E[:, j]
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def backwardE(T, E, n_clones, x, a):
    """
    Compute backward messages with explicit emission matrix.
    
    Parameters
    ----------
    T : numpy.ndarray
        Transition matrix, shape (n_actions, n_states, n_states).
    E : numpy.ndarray
        Emission probability matrix, shape (n_states, n_emissions).
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
        
    Returns
    -------
    mess_bwd : numpy.ndarray
        Backward messages.
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T.dtype.type

    # backward pass
    t = x.shape[0] - 1
    message = np.ones(E.shape[0], dtype)
    message /= message.sum()
    mess_bwd = np.empty((len(x), E.shape[0]), dtype=dtype)
    mess_bwd[t] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, j = (
            a[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        message = T[aij].dot(message * E[:, j])
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        mess_bwd[t] = message
    return mess_bwd


@nb.njit
def updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a):
    """
    Update transition matrix counts using forward-backward messages.
    
    Parameters
    ----------
    C : numpy.ndarray
        Matrix to store transition counts, shape (n_actions, n_states, n_states).
    T : numpy.ndarray
        Current transition probability matrix.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    mess_fwd : numpy.ndarray
        Forward messages from forward algorithm.
    mess_bwd : numpy.ndarray
        Backward messages from backward algorithm.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
        
    Notes
    -----
    Updates C in-place with transition counts based on forward-backward algorithm.
    """
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    timesteps = len(x)
    C[:] = 0
    for t in range(1, timesteps):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (tm1_start, tm1_stop), (t_start, t_stop) = (
            mess_loc[t - 1 : t + 1],
            mess_loc[t : t + 2],
        )
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        q = (
            mess_fwd[tm1_start:tm1_stop].reshape(-1, 1)
            * T[aij, i_start:i_stop, j_start:j_stop]
            * mess_bwd[t_start:t_stop].reshape(1, -1)
        )
        q /= q.sum()
        C[aij, i_start:i_stop, j_start:j_stop] += q


@nb.njit
def forward(T_tr, Pi, n_clones, x, a, store_messages=False):
    """
    Compute log-probability of a sequence, and optionally, messages.
    
    Parameters
    ----------
    T_tr : numpy.ndarray
        Transposed transition matrix, shape (n_actions, n_states, n_states).
    Pi : numpy.ndarray
        Initial state distribution.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
    store_messages : bool, optional
        Whether to store and return forward messages.
        
    Returns
    -------
    log2_lik : numpy.ndarray
        Log2 probability of each observation.
    mess_fwd : numpy.ndarray or None
        Forward messages if store_messages is True, None otherwise.
    """
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    j_start, j_stop = state_loc[j : j + 2]
    message = Pi[j_start:j_stop].copy().astype(dtype)
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_loc = np.hstack(
            (np.array([0], dtype=n_clones.dtype), n_clones[x])
        ).cumsum()
        mess_fwd = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    for t in range(1, x.shape[0]):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        message = np.ascontiguousarray(T_tr[aij, j_start:j_stop, i_start:i_stop]).dot(
            message
        )
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd[t_start:t_stop] = message
    return log2_lik, mess_fwd


@nb.njit
def backward(T, n_clones, x, a):
    """
    Backward algorithm for computing backward messages.

    Parameters
    ----------
    T : numpy.ndarray
        Transition matrix, shape (n_actions, n_states, n_states).
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.

    Returns
    -------
    mess_bwd : numpy.ndarray
        Backward messages.
    """
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T.dtype.type

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    message = np.ones(n_clones[i], dtype) / n_clones[i]
    message /= message.sum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    mess_bwd = np.empty(mess_loc[-1], dtype)
    t_start, t_stop = mess_loc[t : t + 2]
    mess_bwd[t_start:t_stop] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        message = np.ascontiguousarray(T[aij, i_start:i_stop, j_start:j_stop]).dot(
            message
        )
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        t_start, t_stop = mess_loc[t : t + 2]
        mess_bwd[t_start:t_stop] = message
    return mess_bwd


@nb.njit
def forward_mp(T_tr, Pi, n_clones, x, a, store_messages=False):
    """
    Max-product forward algorithm for computing most likely sequence.

    Log-probability of a sequence, and optionally, messages
    
    Parameters
    ----------
    T_tr : numpy.ndarray
        Transposed transition matrix, shape (n_actions, n_states, n_states).
    Pi : numpy.ndarray
        Initial state distribution.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
    store_messages : bool, optional
        Whether to store and return forward messages.
        
    Returns
    -------
    log2_lik : numpy.ndarray
        Log2 probability of each observation.
    mess_fwd : numpy.ndarray or None
        Forward messages if store_messages is True, None otherwise.
        
    Notes
    -----
    Uses max instead of sum operations for Viterbi decoding.
    """
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    j_start, j_stop = state_loc[j : j + 2]
    message = Pi[j_start:j_stop].copy().astype(dtype)
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_loc = np.hstack(
            (np.array([0], dtype=n_clones.dtype), n_clones[x])
        ).cumsum()
        mess_fwd = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    for t in range(1, x.shape[0]):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        new_message = np.zeros(j_stop - j_start, dtype=dtype)
        for d in range(len(new_message)):
            new_message[d] = (T_tr[aij, j_start + d, i_start:i_stop] * message).max()
        message = new_message
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd[t_start:t_stop] = message
    return log2_lik, mess_fwd


@nb.njit
def rargmax(x):
    """
    Return a random index corresponding to the maximum value.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array to find maximum value in.
        
    Returns
    -------
    int
        Random index where x has its maximum value.
        
    Notes
    -----
    Used for breaking ties randomly in Viterbi decoding.
    """
    # return x.argmax()  # <- favors clustering towards smaller state numbers
    return np.random.choice((x == x.max()).nonzero()[0])


@nb.njit
def backtrace(T, n_clones, x, a, mess_fwd):
    """
    Backtrace algorithm for maximum a posteriori state sequence.
    
    Parameters
    ----------
    T : numpy.ndarray
        Transition matrix, shape (n_actions, n_states, n_states).
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
    mess_fwd : numpy.ndarray
        Forward messages from max-product forward algorithm.
        
    Returns
    -------
    states : numpy.ndarray
        Most likely state sequence.
    """
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    code = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    t_start, t_stop = mess_loc[t : t + 2]
    belief = mess_fwd[t_start:t_stop]
    code[t] = rargmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), j_start = state_loc[i : i + 2], state_loc[j]
        t_start, t_stop = mess_loc[t : t + 2]
        belief = (
            mess_fwd[t_start:t_stop] * T[aij, i_start:i_stop, j_start + code[t + 1]]
        )
        code[t] = rargmax(belief)
    states = state_loc[x] + code
    return states


def backtraceE(T, E, n_clones, x, a, mess_fwd):
    """
    Backtrace with explicit emission matrix for MAP state sequence.
    
    Parameters
    ----------
    T : numpy.ndarray
        Transition matrix, shape (n_actions, n_states, n_states).
    E : numpy.ndarray
        Emission probability matrix, shape (n_states, n_emissions).
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
    mess_fwd : numpy.ndarray
        Forward messages from max-product forward algorithm.
        
    Returns
    -------
    states : numpy.ndarray
        Most likely state sequence.
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    states = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    belief = mess_fwd[t]
    states[t] = rargmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij = a[t]  # at time t -> t+1 we go from observation i to observation j
        belief = mess_fwd[t] * T[aij, :, states[t + 1]]
        states[t] = rargmax(belief)
    return states


def forwardE_mp(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """
    Max-product forward algorithm with explicit emission matrix.
    
    Parameters
    ----------
    T_tr : numpy.ndarray
        Transposed transition matrix, shape (n_actions, n_states, n_states).
    E : numpy.ndarray
        Emission probability matrix, shape (n_states, n_emissions).
    Pi : numpy.ndarray
        Initial state distribution.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    x : numpy.ndarray
        Observation sequence.
    a : numpy.ndarray
        Action sequence.
    store_messages : bool, optional
        Whether to store and return forward messages.
        
    Returns
    -------
    log2_lik : numpy.ndarray
        Log2 probability of each observation.
    mess_fwd : numpy.ndarray, optional
        Forward messages if store_messages is True.
        
    Notes
    -----
    Uses max instead of sum operations for Viterbi decoding.
    """
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = (T_tr[aij] * message.reshape(1, -1)).max(1)
        message *= E[:, j]
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def forward_mp_all(T_tr, Pi_x, Pi_a, n_clones, target_state, max_steps):
    """
    Max-product forward algorithm for finding paths to a target state.
    
    Parameters
    ----------
    T_tr : numpy.ndarray
        Transposed transition matrix, shape (n_actions, n_states, n_states).
    Pi_x : numpy.ndarray
        Initial state distribution.
    Pi_a : numpy.ndarray
        Action probability distribution.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    target_state : int
        Target state to reach.
    max_steps : int
        Maximum number of steps allowed to reach target state.
        
    Returns
    -------
    log2_lik : numpy.ndarray
        Log2 probability of each step.
    mess_fwd : numpy.ndarray
        Forward messages for each step.
        
    Raises
    ------
    AssertionError
        If unable to find a path to target state in max_steps.
        
    Notes
    -----
    Used for finding optimal paths between states.
    """
    # forward pass
    t, log2_lik = 0, []
    message = Pi_x
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik.append(np.log2(p_obs))
    mess_fwd = []
    mess_fwd.append(message)
    T_tr_maxa = (T_tr * Pi_a.reshape(-1, 1, 1)).max(0)
    for t in range(1, max_steps):
        message = (T_tr_maxa * message.reshape(1, -1)).max(1)
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik.append(np.log2(p_obs))
        mess_fwd.append(message)
        if message[target_state] > 0:
            break
    else:
        assert False, "Unable to find a bridging path"
    return np.array(log2_lik), np.array(mess_fwd)


def backtrace_all(T, Pi_a, n_clones, mess_fwd, target_state):
    """
    Backtrace algorithm for finding optimal actions and states.
    
    Parameters
    ----------
    T : numpy.ndarray
        Transition matrix, shape (n_actions, n_states, n_states).
    Pi_a : numpy.ndarray
        Action probability distribution.
    n_clones : numpy.ndarray
        Number of clones for each observation type.
    mess_fwd : numpy.ndarray
        Forward messages from max-product forward algorithm.
    target_state : int
        Target state that was reached.
        
    Returns
    -------
    actions : numpy.ndarray
        Optimal action sequence.
    states : numpy.ndarray
        Optimal state sequence.
        
    Notes
    -----
    Used with forward_mp_all to determine optimal paths.
    """
    states = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    actions = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    n_states = T.shape[1]
    # backward pass
    t = mess_fwd.shape[0] - 1
    actions[t], states[t] = (
        -1,
        target_state,
    )  # last actions is irrelevant, use an invalid value
    for t in range(mess_fwd.shape[0] - 2, -1, -1):
        belief = (
            mess_fwd[t].reshape(1, -1) * T[:, :, states[t + 1]] * Pi_a.reshape(-1, 1)
        )
        a_s = rargmax(belief.flatten())
        actions[t], states[t] = a_s // n_states, a_s % n_states
    return actions, states
