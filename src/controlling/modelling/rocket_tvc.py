"""
Rocket TVC pitch-axis models.

Each model builder is kept next to its runtime linearizer: they encode the
same state-space structure, and must change together. The builders produce
the nominal ModelConfig (used once, e.g. to compute the steady-state KF gain);
the linearizers rebuild Phi/Gamma each control step as thrust and mass change
during the burn.
"""
import numpy as np

from src.plant.parameters import ActuatorParams, GyroParams
from src.plant.sensors.imu import noise_std
from src.controlling.modelling.modelconfig import ModelConfig


# ===== MPC model: [theta, theta_dot, delta] (+ wind via add_disturbance_state) =====

def ActuatorDelay_RocketTVC(dt, actuator: ActuatorParams):
    unit = "rad"
    K_GAIN = 1.0 # No external scaling needed for now
    omega_c = actuator.omega_c

    c1 = np.exp(-dt * omega_c)
    c2 = 1.0 - c1

    # States: [theta, theta_dot, delta]     delta_dot = omega_c * (delta_cmd - delta)
    Phi = np.array([
        [1.0, dt , 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, c1 ]
    ])

    # Placeholder Gamma (will be overwritten by the controller dynamically)
    Gamma = np.array([
        [0.0],
        [0.0],
        [c2 ]
    ])

    G = np.array([
        [0.5 * dt**2],
        [dt         ],
        [0.0        ]
    ])

    Cy = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    Cz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    var_model = 1e-2
    R1 = var_model * np.eye(1)

    var_measurement = 1e-4
    R2 = var_measurement * np.eye(2)

    model = ModelConfig(
        Phi=Phi, Gamma=Gamma, G=G, Cy=Cy, Cz=Cz,
        R1=R1, R2=R2, K_GAIN=K_GAIN, unit=unit,
    )
    return model

def linearize_actuator_delay(dt, actuator: ActuatorParams, thrust, gimbal_arm, inertia):
    """
    Time-varying Phi, Gamma for the ActuatorDelay_RocketTVC structure,
    linearized about the current operating point (thrust and mass properties
    change during the burn, so this must be recomputed every control step).
    """
    alpha = (-thrust * gimbal_arm) / inertia

    c1 = np.exp(-dt * actuator.omega_c)
    c2 = 1.0 - c1

    Phi = np.array([
        [1.0, dt , 0.5 * alpha * dt**2],
        [0.0, 1.0, alpha * dt         ],
        [0.0, 0.0, c1                 ]
    ])

    Gamma = np.array([
        [0.0],
        [0.0],
        [c2 ]
    ])

    return Phi, Gamma


# ===== Attitude-KF model: [theta_dot, delta, wind] (theta dead-reckoned outside) =====

def IMU_KF_RocketTVC(dt, actuator: ActuatorParams, gyro: GyroParams,
                     var_model=1e-2, var_wind=1e-1, thrust_nom=15.0,
                     gimbal_arm_nom=0.5, inertia_nom=(1/12)*0.925):
    """
    Reduced observable model for the rate-gyro attitude KF: state [theta_dot, delta, wind].

    theta is deliberately NOT in this model — with a rate-only measurement it is
    unobservable (the DARE has no finite solution), and it is fully decoupled from
    the dynamics, so the AttitudeEstimator dead-reckons it separately. What remains
    here is observable (rank 3/3) and the steady-state DARE solves cleanly.

    The measurement is the bias-corrected gyro, treated as a direct reading of
    theta_dot; R2 is the gyro's own noise variance, so nothing is faked. Phi is
    time-varying (alpha depends on thrust/mass); nominal values here are only used
    to compute the steady-state gain, then Phi is replaced each step at runtime.
    """
    Phi, Gamma = linearize_imu_kf(dt, actuator, thrust_nom, gimbal_arm_nom, inertia_nom)

    # process noise: angular-accel uncertainty on theta_dot, plus a wind random walk
    G = np.array([
        [dt , 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])
    R1 = np.diag([var_model, var_wind])

    Cy = np.array([[1.0, 0.0, 0.0]])   # gyro measures theta_dot
    Cz = Cy                            # unused by the KF, kept for ModelConfig

    R2 = np.array([[noise_std(gyro.noise_density, gyro.sample_rate) ** 2]])

    return ModelConfig(Phi=Phi, Gamma=Gamma, G=G, Cy=Cy, Cz=Cz, R1=R1, R2=R2, unit="rad/s")

def linearize_imu_kf(dt, actuator: ActuatorParams, thrust, gimbal_arm, inertia):
    """Time-varying [theta_dot, delta, wind] matrices for the attitude KF (runtime)."""
    alpha = (-thrust * gimbal_arm) / inertia
    c1 = np.exp(-dt * actuator.omega_c)
    c2 = 1.0 - c1

    Phi = np.array([
        [1.0, alpha * dt, dt ],   # theta_dot += alpha*dt*delta + dt*wind
        [0.0, c1        , 0.0],   # delta   (first-order servo)
        [0.0, 0.0       , 1.0],   # wind    (random walk)
    ])
    Gamma = np.array([
        [0.0],
        [c2 ],
        [0.0],
    ])
    return Phi, Gamma
