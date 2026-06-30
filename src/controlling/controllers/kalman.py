import numpy as np
from scipy.linalg import solve_discrete_are, LinAlgError
from src.controlling.modelling.models import ModelConfig

class KalmanFilter:
    def __init__(self, model: ModelConfig):
        self.model = model
        
        self.Phi = model.Phi
        self.Gamma = model.Gamma
        self.G = model.G
        self.Cy = model.Cy
        self.R1 = model.R1
        self.R2 = model.R2
        
        self.L, self.P = self._get_kalman_gain(self.Phi, self.G, self.Cy, self.R1, self.R2)
        
        self.x = np.zeros((model.nbr_states,1)) # x = (px, vx, py, vy)^T, dx = Phi @ x + Gamma @ u.    
    
    def _get_kalman_gain(self, Phi, G, C, R1, R2):
        Q = G @ R1 @ G.T
        try:
            P_inf = solve_discrete_are(Phi.T, C.T, Q, R2) #Prediction variance
            S = C @ P_inf @ C.T + R2
            L_inf = P_inf @ C.T @ np.linalg.inv(S) #Kalman filter gain
            return L_inf, P_inf
        
        except LinAlgError:
            print('Using itterative solver (likely unobservable, ARE fails)')
            P_pred = np.eye(self.model.nbr_states)
            I = np.eye(self.model.nbr_states)
            
            tolerance = 1e-6
            max_iter = 50000
            
            for i in range(max_iter):
                S = C @ P_pred @ C.T + R2
                L = P_pred @ C.T @ np.linalg.inv(S)
                
                P_filt = (I - L @ C) @ P_pred
                P_pred_next = Phi @ P_filt @ Phi.T + Q
                
                diff = np.max(np.abs(P_pred_next - P_pred))
                if diff < tolerance:
                    print(f"Iterative ARE solver converged in {i+1} iterations.")
                    P_pred = P_pred_next
                    break
                P_pred = P_pred_next
                
            if diff > tolerance:
                print(f"WARNING: Iterative solver reached max_iter ({max_iter}) without converging")
                P_pred = P_pred_next
                
            return L, P_pred
            
        
    def predict_step(self, u):
        # x_{k|k-1} = Phi * x_{k-1|k-1} + Gamma * u
        u = np.asarray(u).reshape(-1, 1)
        self.x = self.Phi @ self.x + self.Gamma @ u
            
        return self.x
            
    def filter_step(self, y_measured):
        # x_{k|k} = x_{k|k-1} + L * (y - C * x_{k|k-1})
        innovation = y_measured - self.Cy @ self.x
        self.x = self.x + self.L @ innovation
            
        return self.x
    
    def initialize_state(self, y0, y1, dt):
        y0 = np.asarray(y0).flatten()
        y1 = np.asarray(y1).flatten()
        
        x_init = np.zeros((self.model.nbr_states, 1))
        
        for i in range(self.model.nbr_outputs):
            x_init[i, 0] = y1[i]
            
        self.x = x_init

        
        