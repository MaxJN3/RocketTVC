import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    #Dynamics
    Phi: np.ndarray
    Gamma: np.ndarray
    G: np.ndarray
    Cy: np.ndarray
    Cz: np.ndarray
    #Covariance
    R1: np.ndarray  #covariance w_k (model)
    R2: np.ndarray  #covariance v_k (measurement)
    #Deadband threshold
    deadband_threshold: Optional[np.ndarray] = 0

    unmeasured_states: Optional[list] = None

    #Hardware gain (unit / rad)
    K_GAIN: float = 1
    unit: str = "rad/s"

    @property
    def nbr_states(self):
        return self.Phi.shape[1]
    @property
    def nbr_inputs(self):
        return self.Gamma.shape[1]
    @property
    def nbr_outputs(self):
        return self.Cy.shape[0]
    @property
    def nbr_performance_states(self):
        return self.Cz.shape[0]
