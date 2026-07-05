import numpy as np
from dataclasses import dataclass, field


@dataclass
class MotorParams:
    """Motor properties. Swap these for real data (e.g. thrustcurve.org) once a motor is chosen."""
    burn_time: float = 5.0          # s
    average_thrust: float = 15.0    # N
    propellant_mass: float = 0.15   # kg

    def thrust(self, t):
        """Thrust in Newtons at time t since ignition.

        Constant profile for now; replace with an interpolated thrust curve
        when real motor data is available.
        """
        return self.average_thrust if t < self.burn_time else 0.0

    def propellant_remaining(self, t):
        """Remaining propellant mass in kg, assuming a linear burn."""
        if t >= self.burn_time:
            return 0.0
        return self.propellant_mass * (1.0 - t / self.burn_time)


@dataclass
class AirframeParams:
    dry_mass: float = 0.85       # kg, everything except propellant
    length: float = 1.0          # m
    radius: float = 0.04         # m
    cp_from_nose: float = 0.4    # m, center of pressure measured from the nose


@dataclass
class ActuatorParams:
    """TVC gimbal servo. omega_c is a guess until measured on the real servo."""
    omega_c: float = 20.0                        # rad/s, first-order servo bandwidth
    max_gimbal_rad: float = np.deg2rad(5.0)      # rad, mechanical gimbal limit
    max_gimbal_step: float = 0.05                # rad, max command change per control step


@dataclass
class AeroParams:
    cd: float = 0.5             # drag coefficient (generic tube)
    cna: float = 2.0            # normal force coefficient derivative, per rad (finless tube)
    air_density: float = 1.225  # kg/m^3 at sea level


@dataclass
class GyroParams:
    """Placeholder numbers from the MPU-6050 datasheet — replace with the real IMU's."""
    noise_density: float = np.deg2rad(0.005)   # (rad/s)/sqrt(Hz)
    turn_on_bias_max: float = np.deg2rad(1.0)  # rad/s, drawn uniformly at power-on
    sample_rate: float = 50.0                  # Hz, matches the control rate for v1


@dataclass
class AccelParams:
    """Placeholder numbers from the MPU-6050 datasheet — replace with the real IMU's."""
    noise_density: float = 400e-6 * 9.81       # (m/s^2)/sqrt(Hz)  (400 ug/sqrt(Hz))
    turn_on_bias_max: float = 0.05 * 9.81      # m/s^2  (~50 mg)
    sample_rate: float = 50.0                  # Hz, matches the control rate for v1


@dataclass
class ImuParams:
    gyro: GyroParams = field(default_factory=GyroParams)
    accel: AccelParams = field(default_factory=AccelParams)


@dataclass
class VehicleParams:
    motor: MotorParams = field(default_factory=MotorParams)
    airframe: AirframeParams = field(default_factory=AirframeParams)
    actuator: ActuatorParams = field(default_factory=ActuatorParams)
    aero: AeroParams = field(default_factory=AeroParams)
    imu: ImuParams = field(default_factory=ImuParams)
