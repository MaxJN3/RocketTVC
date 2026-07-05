import numpy as np

from src.plant.parameters import ImuParams

GRAVITY = 9.81


def noise_std(noise_density, sample_rate):
    """Per-sample noise standard deviation from a datasheet noise density.

    White noise with density n sampled at fs captures the band up to
    Nyquist (fs / 2), so sigma = n * sqrt(fs / 2).
    """
    return noise_density * np.sqrt(sample_rate / 2.0)


class Imu:
    """
    Simulated strapdown IMU for the 2D planar sim.

    Produces only what a real IMU produces:
      - gyro: body pitch rate (rad/s)
      - accelerometer: specific force in body axes (m/s^2) — NOT orientation

    Turn-on biases are drawn once at construction ("power-on"), so every
    flight has a different, constant, unknown bias — exactly the error the
    estimator has to deal with on the real vehicle.
    """

    def __init__(self, params: ImuParams):
        self.params = params

        b_a = params.accel.turn_on_bias_max
        self.accel_bias_long = np.random.uniform(-b_a, b_a)
        self.accel_bias_lat = np.random.uniform(-b_a, b_a)

        b_g = params.gyro.turn_on_bias_max
        self.gyro_bias = np.random.uniform(-b_g, b_g)

    def measure(self, theta, theta_dot, ax, ay):
        """
        Args:
            theta:     true pitch angle (rad)
            theta_dot: true pitch rate (rad/s)
            ax, ay:    true inertial acceleration of the vehicle (m/s^2),
                       gravity included (i.e. sitting on the pad: ax = ay = 0)

        Returns:
            dict with gyro rate (rad/s) and specific force f_long, f_lat (m/s^2)
        """
        f_long, f_lat = self._measure_accel(theta, ax, ay)
        return {
            "gyro": self._measure_gyro(theta_dot),
            "f_long": f_long,
            "f_lat": f_lat,
        }

    def _measure_gyro(self, theta_dot):
        """
        Gyro measures body pitch rate: omega_meas = theta_dot + bias + noise.
        The turn-on bias is constant per flight (drawn in __init__); the noise
        is white with per-sample sigma from the datasheet noise density.
        """
        sigma = noise_std(self.params.gyro.noise_density, self.params.gyro.sample_rate)
        noise = np.random.normal(0.0, sigma)
        return theta_dot + self.gyro_bias + noise

    def _measure_accel(self, theta, ax, ay):
        """
        Accelerometers measure specific force: f = a - g, expressed in body axes.

        Inertial frame: X downrange, Y up, gravity vector g_vec = (0, -g).
        Body axes at pitch theta (theta = 0 pointing straight up, positive
        theta tips the nose toward +X — same convention as Kinematics):
            longitudinal axis (toward nose): ( sin(theta),  cos(theta) )
            lateral axis:                    ( cos(theta), -sin(theta) )

        Sanity checks this geometry must satisfy (good unit tests):
          - at rest on the pad (a = 0, theta = 0): f_long = +g, f_lat = 0
            (the accelerometer 'feels' the ground pushing up)
          - pure axial thrust at any theta: f_long = T/m, f_lat = 0
        """
        f_x = ax
        f_y = ay + GRAVITY

        f_long = np.sin(theta) * f_x + np.cos(theta) * f_y
        f_lat = np.cos(theta) * f_x - np.sin(theta) * f_y

        sigma = noise_std(self.params.accel.noise_density, self.params.accel.sample_rate)
        f_long += self.accel_bias_long + np.random.normal(0.0, sigma)
        f_lat += self.accel_bias_lat + np.random.normal(0.0, sigma)

        return f_long, f_lat
