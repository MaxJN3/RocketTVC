import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.plant.parameters import VehicleParams
from src.controlling.controllers.controller import Controller
from src.controlling.modelling.configs import SIM_RocketTVC as cfg_SIM_RocketTVC
from src.controlling.modelling.models import (
    ActuatorDelay_RocketTVC,
    add_disturbance_state,
    linearize_actuator_delay,
    extend_matrices_with_disturbance,
)
from src.controlling.modelling.objectives import MinimizeError
from src.plant.vehicle.rocket import Rocket
from src.plant.vehicle.kinematics import Kinematics
from src.plant.vehicle.aerodynamics import Aerodynamics
from src.plant.logger import FlightLogger
from src.plant.diffeqsolvers import rk4_step


# ===== Scenario =====
dt = 1 / 50
t_end = 5.0

x0 = np.array([0.1, 0.0, 0.0])  # [theta, theta_dot, delta]: launch tipped 0.1 rad

def wind(t):
    """Wind velocity (x, y) in m/s: a 10 m/s crosswind gust hitting at t = 2s."""
    return (-10.0, 0.0) if t >= 2.0 else (0.0, 0.0)


# ===== Plant (physical truth) =====
params = VehicleParams()

rocket = Rocket(params)
kinematics = Kinematics(is_3d=False)
aerodynamics = Aerodynamics(params.airframe, params.aero)

# ===== Controller =====
model_rocket = ActuatorDelay_RocketTVC(dt, params.actuator)
extended_model = add_disturbance_state(model_rocket)
cfg_rocket = cfg_SIM_RocketTVC(params.actuator)

controller = Controller(extended_model, cfg_rocket, MinimizeError(nbr_states=2), dt)

logger = FlightLogger(u_limit_deg=np.degrees(params.actuator.max_gimbal_rad))


x_current = x0
u_opt = np.array([0.0])

for t in np.arange(0, t_end, dt):
    rocket.update_state(t)
    pitch = x_current[0]

    wind_x, wind_y = wind(t)
    aero = aerodynamics.calculate_forces_and_torque(
        kinematics.vx, kinematics.vy, pitch, rocket.cg, wind_x, wind_y
    )

    kinematics.step(dt, thrust=rocket.thrust, mass=rocket.mass, pitch_angle=pitch,
                    fx_aero=aero["F_aero_x"], fy_aero=aero["F_aero_y"])

    logger.log_step(t, x_current, u_opt, pos=kinematics.pos, vel=kinematics.vel)

    # Controller's internal model, re-linearized about the current thrust and mass
    Phi, Gamma = linearize_actuator_delay(dt, params.actuator,
                                          rocket.thrust, rocket.gimbal_arm, rocket.inertia)
    Phi_extended, Gamma_extended = extend_matrices_with_disturbance(Phi, Gamma, model_rocket.G)

    # Measurement: perfect state + Gaussian noise. This is the seam where the
    # IMU simulator will plug in — real sensors do not measure theta directly.
    y_true = model_rocket.Cy @ x_current.reshape(-1, 1)
    measurement_noise = np.random.multivariate_normal(
        mean=np.zeros(model_rocket.Cy.shape[0]),
        cov=model_rocket.R2
    ).reshape(-1, 1)
    y_measured = y_true + measurement_noise

    u_opt, x_hat = controller.step(y_measured=y_measured,
                                   current_Phi=Phi_extended, current_Gamma=Gamma_extended)

    x_current = rk4_step(rocket.dynamics, t, x_current, dt, u_opt, M_aero=aero["M_aero"])

logger.plot_dashboard()
