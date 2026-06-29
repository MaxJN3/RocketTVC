import numpy as np
from src.controlling.controllers.controller import Controller
from src.controlling.modelling.configs import cfg_SIM_RocketTVC
from src.controlling.modelling.models import model_SIM_RocketTVC
from src.controlling.modelling.objectives import MinimizeError

from rocket import Rocket
from diffeqsolvers import rk4_step


dt = 1/50
t_end = 5.0
tt = np.arange(0, t_end, dt)

x_current = np.array([0.1, 0.0])
u_current = np.array([0.0])

rocket = Rocket()
controller = Controller(model_SIM_RocketTVC, cfg_SIM_RocketTVC, MinimizeError, dt)


for t in tt:
    rocket.update_state(t)

    m = rocket.mass
    I = rocket.inertia
    T = rocket.thrust
    l_cg = rocket.cg_distance
    
    B = np.array([
        [0.0],
        [(-T * l_cg) / I]
    ])
    Gamma = B * dt

    u_current = Controller.step(
        y_measured=x_current, # Bypassing kalman noise for the first test
        ref=[0.0, 0.0],       # We want the rocket perfectly upright
        u_prev=u_current,
        current_Gamma=Gamma
    )
    
    x_current = rk4_step(rocket.dynamics, t, x_current, u_current, dt)
    

