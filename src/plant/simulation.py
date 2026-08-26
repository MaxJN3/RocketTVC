import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.plant.parameters import VehicleParams
from src.plant.vehicle.rocket import Rocket
from src.plant.vehicle.kinematics import Kinematics
from src.plant.vehicle.aerodynamics import Aerodynamics
from src.plant.sensors.imu import Imu, GRAVITY
from src.plant.logger import FlightLogger, Panel
from src.plant.diffeqsolvers import rk4_step

from src.controlling.controllers.mpc import ModelPredictiveControl
from src.controlling.estimators.attitude_estimator import AttitudeEstimator
from src.controlling.modelling.configs import SIM_RocketTVC as cfg_SIM_RocketTVC
from src.controlling.modelling.rocket_tvc import (
    ActuatorDelay_RocketTVC,
    linearize_actuator_delay,
    IMU_KF_RocketTVC,
    linearize_imu_kf,
)
from src.controlling.modelling.transforms import (
    add_disturbance_state,
    extend_matrices_with_disturbance,
)
from src.controlling.modelling.objectives import MinimizeError


# ===== Scenario =====
dt = 1 / 50
t_end = 5.0
pad_seconds = 2.0                    # clamped on the pad, calibrating the gyro bias

x0 = np.array([0.1, 0.0, 0.0])       # true [theta, theta_dot, delta]: launch tipped 0.1 rad
launch_angle = x0[0]                 # known from the launch rail

def wind(t):
    """Wind velocity (x, y) in m/s: a 3 m/s crosswind gust hitting at t = 2s."""
    return (-3.0, 0.0) if t >= 2.0 else (0.0, 0.0)


# ===== Plant (physical truth) =====
params = VehicleParams()

rocket = Rocket(params)
kinematics = Kinematics(is_3d=False)
aerodynamics = Aerodynamics(params.airframe, params.aero)
imu = Imu(params.imu)

# ===== Controller: MPC on the full state, KF on the observable subsystem =====
mpc_base = ActuatorDelay_RocketTVC(dt, params.actuator)   # [theta, theta_dot, delta]
mpc_model = add_disturbance_state(mpc_base)               # + wind -> full 4-state
mpc = ModelPredictiveControl(mpc_model, cfg_SIM_RocketTVC(params.actuator))
objective = MinimizeError(nbr_states=2)

kf_model = IMU_KF_RocketTVC(dt, params.actuator, params.imu.gyro)   # [theta_dot, delta, wind]
estimator = AttitudeEstimator(kf_model, dt, launch_angle=launch_angle)

logger = FlightLogger()


# ===== Pad phase: rocket clamped, motor off, true rate = 0 =====
n_pad = int(pad_seconds / dt)
gyro_pad = []
for i in range(n_pad):
    g = imu.measure(theta=launch_angle, theta_dot=0.0, ax=0.0, ay=0.0)["gyro"]
    gyro_pad.append(g)
    logger.log((i - n_pad) * dt, gyro=g)   # negative time: before ignition
estimator.calibrate_bias(gyro_pad)


# ===== Flight =====
x_current = x0.copy()
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

    # True vehicle acceleration, fed to the IMU (Option 1 reads only the gyro,
    # but the accelerometer channel is driven honestly for future use / logging).
    ax_true = (rocket.thrust / rocket.mass) * np.sin(pitch) + aero["F_aero_x"] / rocket.mass
    ay_true = (rocket.thrust / rocket.mass) * np.cos(pitch) - GRAVITY + aero["F_aero_y"] / rocket.mass
    meas = imu.measure(theta=x_current[0], theta_dot=x_current[1], ax=ax_true, ay=ay_true)

    # Re-linearize both models about the current thrust/mass operating point
    Phi_full, Gamma_full = extend_matrices_with_disturbance(
        *linearize_actuator_delay(dt, params.actuator,
                                  rocket.thrust, rocket.gimbal_arm, rocket.inertia),
        mpc_base.G)
    Phi_red, Gamma_red = linearize_imu_kf(dt, params.actuator,
                                          rocket.thrust, rocket.gimbal_arm, rocket.inertia)

    # Estimate -> control -> propagate estimator with the fresh command
    x_hat = estimator.update(meas["gyro"])
    ref = objective.compute_reference(x_hat)
    u_opt = mpc.step(x_hat, ref, u_opt, current_Phi=Phi_full, current_Gamma=Gamma_full)
    estimator.predict(u_opt, Phi_red, Gamma_red)

    logger.log(t,
        # plant truth (pre-propagation, the state the controller acted on)
        theta_true=x_current[0], theta_dot_true=x_current[1], delta_true=x_current[2],
        pos_x=kinematics.x, pos_y=kinematics.y, vx=kinematics.vx, vy=kinematics.vy,
        alpha_aero=aero.get("alpha", 0.0),   # absent below the aero model speed cutoff
        # sensing and estimation
        gyro=meas["gyro"], f_long=meas["f_long"], f_lat=meas["f_lat"],
        theta_est=x_hat[0, 0], theta_dot_est=x_hat[1, 0],
        delta_est=x_hat[2, 0], wind_est=x_hat[3, 0],
        theta_err=x_hat[0, 0] - x_current[0],
        # control
        u_cmd=u_opt[0],
    )

    x_current = rk4_step(rocket.dynamics, t, x_current, dt, u_opt, M_aero=aero["M_aero"])


# ===== Verification =====
theta_true = logger.array("theta_true")
theta_est = logger.array("theta_est")
print(f"gyro bias:   true {np.degrees(imu.gyro_bias):+.4f} deg/s   "
      f"calibrated {np.degrees(estimator.gyro_bias):+.4f} deg/s   "
      f"residual {np.degrees(imu.gyro_bias - estimator.gyro_bias):+.4f} deg/s")
print(f"theta est:   max |est - true| = {np.degrees(np.abs(theta_est - theta_true).max()):.4f} deg")
print(f"stability:   final theta = {np.degrees(theta_true[-1]):+.4f} deg   "
      f"max |theta| = {np.degrees(np.abs(theta_true).max()):.4f} deg")


# ===== Dashboard =====
gimbal_limit_deg = float(np.degrees(params.actuator.max_gimbal_rad))

dashboard = [
    Panel("Attitude", ["theta_true", "theta_est"], unit="deg",
          labels={"theta_true": "true", "theta_est": "estimate"}),
    Panel("Attitude estimation error", ["theta_err"], unit="deg"),
    Panel("Pitch rate (pad calibration at t < 0)", ["gyro", "theta_dot_true", "theta_dot_est"],
          unit="deg/s",
          labels={"gyro": "gyro (raw)", "theta_dot_true": "true", "theta_dot_est": "estimate"}),
    Panel("Gimbal", ["u_cmd", "delta_true", "delta_est"], unit="deg",
          hlines=(gimbal_limit_deg, -gimbal_limit_deg),
          labels={"u_cmd": "command", "delta_true": "true", "delta_est": "estimate"}),
    Panel("Wind disturbance estimate (KF state)", ["wind_est"], unit="deg/s^2"),
    Panel("Angle of attack", ["alpha_aero"], unit="deg"),
    Panel("Velocity", ["vx", "vy"], unit="m/s",
          labels={"vx": "downrange", "vy": "vertical"}),
    Panel("Flight path", ["pos_x", "pos_y"], kind="xy", unit="m",
          labels={"pos_x": "downrange", "pos_y": "altitude"}),
]

logger.plot(dashboard, title="Rocket TVC simulation — pitch axis")
