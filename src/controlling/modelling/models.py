import numpy as np
from dataclasses import dataclass
from typing import Optional

MODEL = {
    1: "SIM_RocketTVC",
    2: "ActuatorDelay_RocketTVC"
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
    
    if model_name == "ActuatorDelay_RocketTVC":
        return ActuatorDelay_RocketTVC(dt=kwargs.get("dt"))
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def SIM_RocketTVC(dt):
    unit = "rad"
    K_GAIN = 1.0 # No external scaling needed for now
    
    # States: [theta (angle), theta_dot (angular velocity)]
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
    R1 = var_model * np.eye(1)
    
    var_measurement = 1e-4
    R2 = var_measurement * np.eye(2)
    
    model = ModelConfig(
        Phi=Phi, Gamma=Gamma, G=G, Cy=Cy, Cz=Cz, 
        R1=R1, R2=R2, K_GAIN=K_GAIN, unit=unit, 
    )
    return model

def ActuatorDelay_RocketTVC(dt):
    unit = "rad"
    K_GAIN = 1.0 # No external scaling needed for now
    omega_c = 20
    
    c1 = np.exp(-dt * omega_c)
    c2 = 1.0 - c1
    
    # States: [theta, theta_dot, delta]     delta_dot = omega_c * (delta_cmd - delta)
    Phi = np.array([
        [1.0, dt , 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, c1 ]
    ]) 
    
    # Placeholder Gamma (will be overwritten by the controller dynamically)
    Gamma = np.array([
        [0.0],
        [0.0],
        [c2 ]
    ])
    
    G = np.array([
        [0.5 * dt**2],
        [dt         ],
        [0.0        ]
    ])
    
    Cy = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    Cz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    var_model = 1e-2
    R1 = var_model * np.eye(1)
    
    var_measurement = 1e-4
    R2 = var_measurement * np.eye(2)
    
    model = ModelConfig(
        Phi=Phi, Gamma=Gamma, G=G, Cy=Cy, Cz=Cz, 
        R1=R1, R2=R2, K_GAIN=K_GAIN, unit=unit,
    )
    return model

def add_disturbance_state(base_model: ModelConfig, Gammad=None, Cd=None, Cdz=None, var_dist=1e-1):
    """
    Augments a state-space model with a constant disturbance state.
    Defaults to assuming the disturbance is an external acceleration (like wind).
    """
    Phi = base_model.Phi
    Gamma = base_model.Gamma
    G = base_model.G
    Cy = base_model.Cy
    Cz = base_model.Cz
    R1 = base_model.R1
    
    nbr_x = Phi.shape[1]
    nbr_w = G.shape[1]
    
    if Gammad is None:
        Gammad = G
    nbr_d = Gammad.shape[1]
    
    if Cd is None:
        Cd = np.zeros((Cy.shape[0], nbr_d))
    if Cdz is None:
        Cdz = np.zeros((Cz.shape[0], nbr_d))
        
    R1d = var_dist * np.eye(nbr_d)

    Phi_extended = np.block([
        [Phi, Gammad],                            
        [np.zeros((nbr_d, nbr_x)), np.eye(nbr_d)] 
    ])
    
    Gamma_extended = np.vstack([
        Gamma,
        np.zeros((nbr_d, Gamma.shape[1]))
    ])
    
    G_extended = np.block([
        [G, np.zeros((nbr_x, nbr_d))],
        [np.zeros((nbr_d, nbr_w)), np.eye(nbr_d)]
    ])  
    
    Cy_extended = np.hstack([Cy, Cd])
    Cz_extended = np.hstack([Cz, Cdz])
    
    R1_extended = np.block([
        [R1, np.zeros((R1.shape[0], R1d.shape[1]))],
        [np.zeros((R1d.shape[0], R1.shape[1])), R1d]
    ])
    
    return ModelConfig(
        Phi=Phi_extended,
        Gamma=Gamma_extended,
        Cy=Cy_extended,
        Cz=Cz_extended,
        G=G_extended,
        R1=R1_extended,
        R2=base_model.R2,
        deadband_threshold=base_model.deadband_threshold,
        unmeasured_states=base_model.unmeasured_states,
        K_GAIN=base_model.K_GAIN,
        unit=base_model.unit
    )