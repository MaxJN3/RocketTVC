import numpy as np
import cvxpy as cp
from src.controlling.modelling.models import ModelConfig
from src.controlling.modelling.configs import MPCConfig

class ModelPredictiveControl:
    def __init__(self, model: ModelConfig, cfg: MPCConfig):
        self.nbr_states = model.nbr_states
        self.nbr_inputs = model.nbr_inputs
        self.nbr_performance_states = model.nbr_performance_states
        
        self.model = model
        self.cfg = cfg
        
        self.Phi = model.Phi
        self.Gamma = model.Gamma

        self.Q1 = cfg.Q1
        self.Q2 = cfg.Q2
        self.Qf = self.cfg.Qf if self.cfg.Qf is not None else self.Q1

        self.Hp = cfg.Hp
        self.Hu = cfg.Hu
        
        self._validate_dimensions()
        self._build_problem()
        
    def _validate_dimensions(self):
        assert self.Phi.shape == (self.nbr_states, self.nbr_states), "Phi must be (nx, nx)"
        assert self.Gamma.shape[0] == self.nbr_states, "Gamma must have nx rows"
        assert self.Gamma.shape[1] == self.nbr_inputs, "Gamma must have nu columns"
        assert self.Q1.shape == (self.nbr_performance_states, self.nbr_performance_states), "Q1 must be (nz, nz)"
        assert self.Q2.shape == (self.nbr_inputs, self.nbr_inputs), "Q2 must be (nu, nu)"
        
    def _build_problem(self):
        self.x = cp.Variable((self.nbr_states, self.Hp + 1))
        self.u = cp.Variable((self.nbr_inputs, self.Hu))
        self.e = cp.Variable((self.nbr_performance_states, self.Hp + 1))
        
        self.x0_param = cp.Parameter(self.nbr_states)
        self.ref_param = cp.Parameter(self.nbr_performance_states)
        self.u_prev_param = cp.Parameter(self.nbr_inputs)
        
        self.Phi_param = cp.Parameter((self.nbr_states, self.nbr_states))
        self.Gamma_param = cp.Parameter((self.nbr_states, self.nbr_inputs))
        
        self.cost = 0
        self.constraints = []
        self.constraints += [self.x[:,0] == self.x0_param] 
        
        self._add_dynamics_constraints()
        self._add_input_rate_constraints()
        self._add_cost_state()
        self._add_cost_control()
        self._add_bounds_constraints()
        
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        
    def _add_dynamics_constraints(self):
        for k in range(self.Hp):
            u_k = self.u[:, k] if k < self.Hu else self.u[:, self.Hu - 1]
            
            x_next = self.Phi_param @ self.x[:, k] + self.Gamma_param @ u_k
            self.constraints += [self.x[:, k + 1] == x_next]
            
    def _add_input_rate_constraints(self):
        if self.cfg.u_max_step is not None:
            self.constraints += [cp.abs(self.u[:, 0] - self.u_prev_param) <= self.cfg.u_max_step]
            
            for k in range(1, self.Hu):
                self.constraints += [cp.abs(self.u[:, k] - self.u[:, k - 1]) <= self.cfg.u_max_step]
                
    def _add_cost_state(self):
        for k in range(self.Hp):
            self.constraints += [self.e[:, k] == (self.model.Cz @ self.x[:, k] - self.ref_param)]
            self.cost += cp.quad_form(self.e[:, k], self.Q1)
        
        if self.cfg.use_terminal_cost:
            self.constraints += [self.e[:, self.Hp] == (self.model.Cz @ self.x[:, self.Hp] - self.ref_param)]
            self.cost += cp.quad_form(self.e[:, self.Hp], self.Qf)
            
    def _add_cost_control(self):
        for k in range(self.Hu):
            self.cost += cp.quad_form(self.u[:, k], self.Q2)
            
    def _add_bounds_constraints(self):
        if self.cfg.u_min is not None:
            for k in range(self.Hu):
                self.constraints += [self.u[:, k] >= self.cfg.u_min]
                
        if self.cfg.u_max is not None:
            for k in range(self.Hu):
                self.constraints += [self.u[:, k] <= self.cfg.u_max]
        
        if self.cfg.x_min is not None or self.cfg.x_max is not None:
            slack = cp.Variable((self.nbr_states, self.Hp + 1), nonneg=True)
            self.cost += self.cfg.slack_penalty * cp.sum(slack)
            
            if self.cfg.x_min is not None:
                for k in range(self.Hp + 1):
                    self.constraints += [self.x[:, k] >= self.cfg.x_min - slack[:, k]]

            if self.cfg.x_max is not None:
                for k in range(self.Hp + 1):
                    self.constraints += [self.x[:, k] <= self.cfg.x_max + slack[:, k]]
    
    def solve(self, x0, ref, u_prev=None, current_Phi=None, current_Gamma=None):
        self.x0_param.value = np.asarray(x0).flatten()
        self.ref_param.value = np.asarray(ref).flatten()
        
        if u_prev is not None:
            self.u_prev_param.value = np.asarray(u_prev).flatten()
        else:
            self.u_prev_param.value = np.zeros(self.nbr_inputs)
            
        if current_Phi is not None:
            self.Phi_param.value = current_Phi
        else:
            self.Phi_param.value = self.Phi  
            
        if current_Gamma is not None:
            self.Gamma_param.value = current_Gamma
        else:
            self.Gamma_param.value = self.Gamma
        
        self.problem.solve(enforce_dpp=True)
        
        result = {
            "status": self.problem.status,
            "cost": self.problem.value,
            "u0": None,
            "x_pred": None,
            "u_pred": None,
        }
        
        if self.problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            result["u0"] = self.u[:, 0].value
            result["x_pred"] = self.x.value
            result["u_pred"] = self.u.value

        return result
    
    def step(self, x0, ref, u_prev, current_Phi=None, current_Gamma=None):
        result = self.solve(x0, ref, u_prev, current_Phi=current_Phi, current_Gamma=current_Gamma)
        
        if result["u0"] is None:
            raise RuntimeError(f"MPC solve failed with status: {result['status']}")
        
        return result["u0"]
    
    def get_prediction(self):
        if self.x.value is None or self.u.value is None:
            raise RuntimeError("Problem has not been solved yet.")
        return self.x.value, self.u.value
            
    