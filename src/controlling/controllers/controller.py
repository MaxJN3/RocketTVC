import numpy as np
from src.controlling.controllers.kalman import KalmanFilter
from src.controlling.controllers.mpc import ModelPredictiveControl

class Controller:
    def __init__(self, model, cfg, objective, dt=0.1, max_lost_frames=5):
        self.dt = dt
        self.max_lost_frames = max_lost_frames
        
        self.cfg = cfg
        self.model = model
        self.objective = objective
        
        self.mpc = ModelPredictiveControl(self.model, self.cfg)
        self.kf = KalmanFilter(self.model)
        
        self.u_current = np.zeros(self.mpc.nbr_inputs)
        
        self.is_tracking = False
        self.y_prev = None
        self.lost_counter = 0
        
    def step(self, y_measured, current_Phi=None, current_Gamma=None):
        #=====1: WE SEE OBJECT=====#
        if y_measured is not None:
            y_measured = np.asarray(y_measured).reshape(-1, 1)
            
            if not self.is_tracking:    #we are not tracking, have lost sight of object before.
                if self.y_prev is None: #we have no previous measurment, get two in a row and then reinitialize.
                    self.y_prev = y_measured
                    self.lost_counter = 0
                    self.u_current = np.zeros(self.mpc.nbr_inputs)
                    return self.u_current, None
                
                else:                   #we have two consecutive measurements, reinitialize and track the object.
                    saved_states = {}
                    if self.model.unmeasured_states:
                        for idx in self.model.unmeasured_states:
                            saved_states[idx] = float(self.kf.x[idx, 0])
                            
                    self.kf.initialize_state(self.y_prev, y_measured, self.dt)
                    for idx, val in saved_states.items():
                        self.kf.x[idx, 0] = val
                    
                    self.is_tracking = True

            self.lost_counter = 0
            self.y_prev = y_measured
            
            x_hat = self.kf.filter_step(y_measured)
            
            ref = self.objective.compute_reference(x_hat)

            u_opt = self.mpc.step(x_hat, ref, self.u_current, current_Phi=current_Phi, current_Gamma=current_Gamma)

            u_applied = self._compensate_deadband(u_opt)
            
            if current_Gamma is not None:
                self.kf.Gamma = current_Gamma
            self.kf.predict_step(u_applied)
            
            self.u_current = u_applied
            return u_applied, x_hat
        
        #=====2: WE DO NOT SEE THE OBJECT=====#
        else:
            self.lost_counter += 1
            
            if self.is_tracking and self.lost_counter < self.max_lost_frames: #let kalman coast
                x_hat = self.kf.x
                
                ref = self.objective.compute_reference(x_hat)
                
                u_opt = self.mpc.step(x_hat, ref, self.u_current, current_Phi=current_Phi, current_Gamma=current_Gamma)
                
                u_applied = self._compensate_deadband(u_opt)
                
                if current_Gamma is not None:
                    self.kf.Gamma = current_Gamma
                self.kf.predict_step(u_applied)
                
                self.u_current = u_applied
                return u_applied, x_hat
            
            else: #we have lost the object for too many frames. Stop coasting.
                self.is_tracking = False
                self.y_prev = None
                
                self.u_current = np.zeros(self.mpc.nbr_inputs)
                return self.u_current, None
            
    def _compensate_deadband(self, u_opt):
        u_applied = np.array(u_opt, dtype=float).flatten()
        
        noise_floor = 1e-3
        for i in range(self.model.nbr_inputs):
            if abs(u_opt[i]) < noise_floor:
                u_applied[i] = 0
                
            elif u_applied[i] > 0:
                u_applied[i] = max(u_opt[i], self.model.deadband_threshold)
            
            elif u_applied[i] < 0:
                u_applied[i] = min(u_opt[i], -self.model.deadband_threshold)
                
        return u_applied