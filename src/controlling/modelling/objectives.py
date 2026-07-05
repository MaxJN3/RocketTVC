import numpy as np

class Objective:
    def compute_reference(self, x_hat):
            raise NotImplementedError("Each detector must implement compute_reference!")

class MinimizeError(Objective):
    """Objective for minimizing e = r - x. I.e, minimize the error"""
    def __init__(self, nbr_states: int, ref=None):
        self.nbr_states = nbr_states
        self.ref = np.zeros((nbr_states, 1)) if ref is None else ref
        
    def update_target(self, new_ref):
        self.ref = np.asarray(new_ref).reshape(-1 ,1)
        
    def compute_reference(self, x_hat):
        return self.ref
        
        
        