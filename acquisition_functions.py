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

def max_entropy(model, X_pool: torch.Tensor, n_query: int = 10, T: int = 100):
    """Choose pool points that maximise the predictive entropy. Given
    H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)
    
    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise max_entropy a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations
    """
    random_subset = np.random.choice(range(len(X)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(model.estimator.forward(X[random_subset], \
                  training=True),dim=-1).cpu().numpy() for t in range(100)])
    pc = outputs.mean(axis=0)
    acquisition = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def bald(model, X_pool: torch.Tensor, n_query: int = 10, T: int = 100):
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
    random_subset = np.random.choice(range(len(X)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], \
                  training=True),dim=-1).cpu().numpy() for t in range(T)])
    pc = outputs.mean(axis=0)
    H   = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    E_H = - np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)  # [batch size]
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def var_ratios():
    """Like Max Entropy but Variational Ratios measures lack of confidence"""
    pass
