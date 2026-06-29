import numpy as np

OBJECTIVES = {
    1: "CenterCamera",
    2: "PointInTargetDirection"
}

def load_objective(objective_name: str, **kwargs):
    
    if objective_name == "CenterCamera":
        return CenterCamera(nbr_states=kwargs.get("nbr_states", 4))
    
    elif objective_name == "PanInTargetDirection":
        if "fov_degrees" not in kwargs:
            raise ValueError("PanInTargetDirection requires 'fov_degrees'")
        return PanInTargetDirection(fov_degrees=kwargs["fov_degrees"])
        
    elif objective_name == "MinimizeError":
        return MinimizeError(nbr_states=kwargs.get("nbr_states", 2))
    
    else:
        raise ValueError(f"Unknown objective: {objective_name}")

class Objective:
    def compute_reference(self, x_hat):
            raise NotImplementedError("Each detector must implement compute_reference!")

class CenterCamera(Objective):
    """Objective for Pan and or Tilt: keep the target at [0, 0]"""
    def __init__(self, nbr_states=2):
        self.nbr_states = nbr_states
        self.static_ref = np.zeros((self.nbr_states, 1))
        
    def compute_reference(self, x_hat):
        return self.static_ref
    
class PanInTargetDirection(Objective):
    """Objective for stationary camera + pointer motor: track target angle"""
    def __init__(self, fov_degrees: float):
        self.fov = fov_degrees
        self.nbr_states = 1
        
    def compute_reference(self, x_hat):
        # Assumed state vector x = [object_pos (px), object_vel (px), motor_pos (rad)]
        ref = np.zeros((self.nbr_states, 1))

        pos = float(x_hat[0].item())
        target_angle = pos * self.fov / 2
        target_rad = target_angle * (np.pi / 180)
        
        ref[0, 0] = target_rad    
        return ref

class MinimizeError(Objective):
    """Objective for minimizing e = r - x. I.e, minimize the error"""
    def __init__(self, nbr_states: int, ref=None):
        self.nbr_states = nbr_states
        self.ref = np.zeros((nbr_states, 1)) if ref is None else ref
        
    def update_target(self, new_ref):
        self.ref = np.asarray(new_ref).reshape(-1 ,1)
        
    def compute_reference(self, x_hat):
        return self.ref
        
        
        