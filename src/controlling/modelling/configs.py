import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.plant.parameters import ActuatorParams

@dataclass
class MPCConfig:
    Hp: int #prediction horizon
    Hu: int #control horizon
    Q1: np.ndarray  #pentaly x_k
    Q2: np.ndarray  #penalty u_k
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    u_max_step: Optional[np.ndarray] = None
    use_terminal_cost: bool = False
    Qf: Optional[np.ndarray] = None
    slack_penalty: Optional[float] = 10000

def SIM_RocketTVC(actuator: ActuatorParams):
    Hp = 20
    Hu = 20

    Q1 = np.diag([1000.0, 10.0])

    Q2 = np.diag([10.0])

    # Input bounds are the gimbal's physical limits
    u_min = np.array([-actuator.max_gimbal_rad])
    u_max = np.array([actuator.max_gimbal_rad])

    u_max_step = np.array([actuator.max_gimbal_step])
    
    use_terminal_cost = True
    
    config = MPCConfig(
        Hp=Hp, Hu=Hu, Q1=Q1, Q2=Q2, 
        u_min=u_min, u_max=u_max, 
        use_terminal_cost=use_terminal_cost, 
        u_max_step=u_max_step
    )
    return config