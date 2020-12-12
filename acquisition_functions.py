import numpy as np


class AcquisitionFunctions:
    """There are four different acquisition functions:
    Uniform(random), BALD, Max Entropy, Var Ratios"""

    def uniform(self):
        """Baseline acquisition a(x) = unif() with unif() a function
        returning a draw from a uniform distribution over the interval [0,1].
        Using this acquisition function is equivalent to choosing a point
        uniformly at random from the pool."""
        pass
    
    def max_entropy(self):
        """Choose pool points that maximise the predictive entropy. Given
        H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)"""
        pass

    def bald(self):
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

    def var_ratios(self):
        """Like Max Entropy but Variational Ratios measures lack of confidence"""
        pass
