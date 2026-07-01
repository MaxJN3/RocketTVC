import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.controlling.controllers.controller import Controller
from src.controlling.modelling.configs import SIM_RocketTVC as cfg_SIM_RocketTVC
from src.controlling.modelling.models import SIM_RocketTVC as model_SIM_RocketTVC, ActuatorDelay_RocketTVC, add_disturbance_state
from src.controlling.modelling.objectives import MinimizeError

from rocket import Rocket
from logger import FlightLogger
from kinematics import Kinematics
from aerodyanmics import Aerodynamics
from diffeqsolvers import rk4_step


dt = 1/50
t_end = 5.0
tt = np.arange(0, t_end, dt)

x_current = np.array([0.1, 0.0, 0.0])
u_opt = np.array([0.0])

rocket = Rocket()

model_rocket = ActuatorDelay_RocketTVC(dt)
extended_model = add_disturbance_state(model_rocket)
cfg_rocket = cfg_SIM_RocketTVC()

controller = Controller(extended_model, cfg_rocket, MinimizeError(nbr_states=2), dt)

logger = FlightLogger()
kinematics = Kinematics(is_3d=False)
aerodynamics = Aerodynamics(rocket.radius, rocket.cp, cd=0.5, cna=2.0)

for t in tt:
    rocket.update_state(t)

    m = rocket.mass
    I = rocket.inertia
    T = rocket.thrust
    l_cg = rocket.cg
    
    pitch = x_current[0]
    
    wind_x = -10.0 if t >= 2.0 else 0.0
    wind_y = 0.0
    
    aero = aerodynamics.calculate_forces_and_torque(kinematics.vx, kinematics.vy, pitch, l_cg, wind_x, wind_y)
    
    kinematics.step(dt, thrust=T, mass=m, pitch_angle=pitch, fx_aero=aero["F_aero_x"], fy_aero=aero["F_aero_y"])
    
    logger.log_step(t, x_current, u_opt, pos=kinematics.pos, vel=kinematics.vel)
    
    alpha = (-T * l_cg) / I
    omega_c = 20
    c1 = np.exp(-dt * omega_c)
    c2 = 1.0 - c1
    
    Phi = np.array([
        [1.0, dt , 0.5 * alpha * dt**2], 
        [0.0, 1.0, alpha * dt         ],         
        [0.0, 0.0, c1                 ]                  
    ])
    
    Gammad = np.array([[0.5 * dt**2], [dt], [0.0]])
    
    Phi_extended = np.block([
        [Phi, Gammad],
        [np.zeros((1, 3)), np.ones((1, 1))] # Wind stays constant over the step
    ])
    
    Gamma = np.array([
        [0.0],
        [0.0],
        [c2 ],
        [0.0]                          
    ])

    y_true = model_rocket.Cy @ x_current.reshape(-1, 1)
    measurement_noise = np.random.multivariate_normal(
        mean=np.zeros(model_rocket.Cy.shape[0]), 
        cov=model_rocket.R2
    ).reshape(-1, 1)
    y_measured = y_true + measurement_noise
    
    u_opt, x_hat = controller.step(y_measured=y_measured, current_Phi=Phi_extended, current_Gamma=Gamma)
    
    f_rocket = lambda t_sub, x_sub, u_sub: rocket.dynamics(t_sub, x_sub, u_sub, M_aero=aero["M_aero"])
    x_current = rk4_step(f_rocket, t, x_current, u_opt, dt)
    
logger.plot_dashboard()


