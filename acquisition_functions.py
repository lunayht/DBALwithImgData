import numpy as np
import torch

def uniform(X_pool: torch.Tensor, n_query: int = 10):
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

def max_entropy(model, X_pool: torch.Tensor, n_query: int = 10, T: int = 10):
    """Choose pool points that maximise the predictive entropy. Given
    H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)
    
    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations
    """
    pass

def bald():
    """Choose pool points that are expected to maximise the information 
    gained about the model parameters, i.e. maximise the mutal information
    between predictions and model posterior. Given
    I[y,w|x,D_train] = H[y|x,D_train] - E_{p(w|D_train)}[H[y|x,w]]
    with w the model parameters (H[y|x,w] is the entropy of y given w).
    Points that maximise this acquisition function are points on which the
    model is uncertain on average but there exist model parameters that produce
    disagreeing predictions with high certainty. This is equivalent to points
    with high variance in th einput to the softmax layer"""
    pass

def var_ratios():
    """Like Max Entropy but Variational Ratios measures lack of confidence"""
    pass
