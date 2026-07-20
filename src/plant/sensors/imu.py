import numpy as np

from src.plant.parameters import ImuParams

GRAVITY = 9.81


def noise_std(noise_density, sample_rate):
    """Per-sample noise standard deviation from a datasheet noise density"""
    return noise_density * np.sqrt(sample_rate / 2.0)


class Imu:
    """
    Simulated IMU for the 2D planar sim"""

    def __init__(self, params: ImuParams):
        self.params = params

        b_a = params.accel.turn_on_bias_max
        self.accel_bias_long = np.random.uniform(-b_a, b_a)
        self.accel_bias_lat = np.random.uniform(-b_a, b_a)

        b_g = params.gyro.turn_on_bias_max
        self.gyro_bias = np.random.uniform(-b_g, b_g)

    def measure(self, theta, theta_dot, ax, ay):
        f_long, f_lat = self._measure_accel(theta, ax, ay)
        return {
            "gyro": self._measure_gyro(theta_dot),
            "f_long": f_long,
            "f_lat": f_lat,
        }

    def _measure_gyro(self, theta_dot):
        """
        Gyro measures body pitch rate: omega_meas = theta_dot + bias + noise.
        """
        sigma = noise_std(self.params.gyro.noise_density, self.params.gyro.sample_rate)
        noise = np.random.normal(0.0, sigma)
        return theta_dot + self.gyro_bias + noise

    def _measure_accel(self, theta, ax, ay):
        """
        longitudinal axis ( sin(theta),  cos(theta) )
        lateral axis:     ( cos(theta), -sin(theta) )
        """
        f_x = ax
        f_y = ay + GRAVITY

        f_long = np.sin(theta) * f_x + np.cos(theta) * f_y
        f_lat = np.cos(theta) * f_x - np.sin(theta) * f_y

        sigma = noise_std(self.params.accel.noise_density, self.params.accel.sample_rate)
        f_long += self.accel_bias_long + np.random.normal(0.0, sigma)
        f_lat += self.accel_bias_lat + np.random.normal(0.0, sigma)

        return f_long, f_lat
