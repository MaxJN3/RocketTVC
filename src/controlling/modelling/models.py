import numpy as np
from dataclasses import dataclass
from typing import Optional

MODEL = {
    1: "SIM_RocketTVC",
}

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

def load_model(model_name: str, **kwargs):
    
    if model_name == "SIM_RocketTVC":
        return SIM_RocketTVC(dt=kwargs.get("dt"))
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def SIM_RocketTVC(dt):
    unit = "rad"
    K_GAIN = 1.0 # No external scaling needed for now
    
    # States: [theta (angle), theta_dot (angular velocity)]
    unmeasured_states = [] # Assuming IMU gives us both for now
    
    Phi = np.array([
        [1.0, dt ],
        [0.0, 1.0]
    ]) 
    
    # Placeholder Gamma (will be overwritten by the controller dynamically)
    Gamma = np.array([
        [0.0],
        [0.0]
    ])
    
    G = np.array([
        [0.5 * dt**2],
        [dt         ]
    ])
    
    Cy = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    Cz = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    var_model = 1e-2
    R1 = var_model * np.eye(2)
    
    var_measurement = 1e-4
    R2 = var_measurement * np.eye(2)
    
    model = ModelConfig(
        Phi=Phi, Gamma=Gamma, G=G, Cy=Cy, Cz=Cz, 
        R1=R1, R2=R2, K_GAIN=K_GAIN, unit=unit, 
        unmeasured_states=unmeasured_states
    )
    return model