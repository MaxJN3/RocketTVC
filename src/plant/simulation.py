import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.controlling.controllers.controller import Controller
from src.controlling.modelling.configs import SIM_RocketTVC as cfg_SIM_RocketTVC
from src.controlling.modelling.models import SIM_RocketTVC as model_SIM_RocketTVC, add_disturbance_state
from src.controlling.modelling.objectives import MinimizeError

from rocket import Rocket
from logger import FlightLogger
from diffeqsolvers import rk4_step


dt = 1/50
t_end = 5.0
tt = np.arange(0, t_end, dt)

x_current = np.array([0.1, 0.0])
u_opt = np.array([0.0])

rocket = Rocket()

model_rocket = model_SIM_RocketTVC(dt)
extended_model = add_disturbance_state(model_rocket)
cfg_rocket = cfg_SIM_RocketTVC()

controller = Controller(extended_model, cfg_rocket, MinimizeError(nbr_states=2), dt)

logger = FlightLogger()

for t in tt:
    rocket.update_state(t)

    m = rocket.mass
    I = rocket.inertia
    T = rocket.thrust
    l_cg = rocket.cg_distance
    
    logger.log_step(t, x_current, u_opt, T, m)
    
    alpha = (-T * l_cg) / I
    Gamma = np.array([
        [0.5 * alpha * dt**2],
        [alpha * dt],
        [0.0]
    ])

    y_true = model_rocket.Cy @ x_current.reshape(-1, 1)
    measurement_noise = np.random.multivariate_normal(
        mean=np.zeros(model_rocket.Cy.shape[0]), 
        cov=model_rocket.R2
    ).reshape(-1, 1)
    y_measured = y_true + measurement_noise
    
    u_opt, x_hat = controller.step(y_measured=y_measured, current_Gamma=Gamma)
    
    x_current = rk4_step(rocket.dynamics, t, x_current, u_opt, dt)
    
    true_wind_accel = 0.0
    if t >= 2.0:
        true_wind_accel = 2.0
        
    x_current = x_current + (model_rocket.G @ np.array([true_wind_accel])).flatten()
    
logger.plot_dashboard()




