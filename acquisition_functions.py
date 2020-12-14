import torch
import numpy as np
from scipy import stats


def predictions_from_pool(model, X_pool: np.ndarray, T: int = 100, training: bool = True):
    """Run random_subset prediction on model and return the output

    Attributes:
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    random_subset = np.random.choice(range(len(X_pool)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack(
            [
                torch.log_softmax(
                    model.estimator.forward(X_pool[random_subset], training=training),
                    dim=-1,
                )
                .cpu()
                .numpy()
                for t in range(T)
            ]
        )
    return outputs, random_subset


def uniform(model, X_pool: np.ndarray, n_query: int = 10, training: bool = True):
    """Baseline acquisition a(x) = unif() with unif() a function
    returning a draw from a uniform distribution over the interval [0,1].
    Using this acquisition function is equivalent to choosing a point
    uniformly at random from the pool.

    Attributes:
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that randomly select from pool set
    """
    query_idx = np.random.choice(range(len(X_pool)), size=n_query, replace=False)
    return query_idx, X_pool[query_idx]


def shannon_entropy_function(
    model, X_pool: np.ndarray, T: int = 100, E_H: bool = False, training: bool = True
):
    """H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        E_H: If True, compute H and EH for BALD (default: False)
    """
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training=training)
    pc = outputs.mean(axis=0)
    H = (-pc * np.log(pc + 1e-10)).sum(axis=-1)
    if E_H:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E, random_subset
    return H, random_subset


def max_entropy(model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True):
    """Choose pool points that maximise the predictive entropy.
    Using Shannon entropy function.

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise max_entropy a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations
    """
    acquisition, random_subset = shannon_entropy_function(model, X_pool, T, training=training)
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def bald(model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True):
    """Choose pool points that are expected to maximise the information
    gained about the model parameters, i.e. maximise the mutal information
    between predictions and model posterior. Given
    I[y,w|x,D_train] = H[y|x,D_train] - E_{p(w|D_train)}[H[y|x,w]]
    with w the model parameters (H[y|x,w] is the entropy of y given w).
    Points that maximise this acquisition function are points on which the
    model is uncertain on average but there exist model parameters that produce
    disagreeing predictions with high certainty. This is equivalent to points
    with high variance in th einput to the softmax layer

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise bald a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations
    """
    H, E_H, random_subset = shannon_entropy_function(model, X_pool, T, E_H=True, training=training)
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def var_ratios(model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True):
    """Like Max Entropy but Variational Ratios measures lack of confidence.
    Given: variational_ratio[x] := 1 - max_{y} p(y|x,D_{train})

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise var_ratios a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations
    """
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training)
    preds = np.argmax(outputs, axis=2)
    _, count = stats.mode(preds, axis=0)
    acquisition = (1 - count / preds.shape[1]).reshape((-1,))
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]
