import numpy as np

from src.plant.parameters import ActuatorParams, GyroParams
from src.plant.sensors.imu import noise_std
from src.controlling.modelling.modelconfig import ModelConfig


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
    change during the burn, recompute every step).
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
    Reduced observable model for the rate-gyro attitude KF state [theta_dot, delta, wind]"""
    
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
