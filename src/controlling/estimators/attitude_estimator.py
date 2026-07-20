import numpy as np

from src.controlling.controllers.kalman import KalmanFilter


class AttitudeEstimator:
    """
    Produces [theta, theta_dot, delta, wind] for MPC.
    """

    def __init__(self, kf_model, dt, launch_angle=0.0):
        self.dt = dt
        self.kf = KalmanFilter(kf_model)         # state [theta_dot, delta, wind]
        self.kf.x = np.zeros((kf_model.nbr_states, 1))

        self.theta = launch_angle
        self.gyro_bias = 0.0

        self._theta_dot = 0.0

    def calibrate_bias(self, gyro_samples):
        self.gyro_bias = float(np.mean(gyro_samples))
        return self.gyro_bias

    def update(self, gyro):
        """Return [theta, theta_dot, delta, wind]"""
        omega = gyro - self.gyro_bias

        x_red = self.kf.filter_step(np.array([[omega]]))
        self._theta_dot = float(x_red[0, 0])
        delta = float(x_red[1, 0])
        wind = float(x_red[2, 0])

        return np.array([[self.theta], [self._theta_dot], [delta], [wind]])

    def predict(self, u, Phi_reduced, Gamma_reduced):
        self.kf.Phi = Phi_reduced
        self.kf.Gamma = Gamma_reduced
        self.kf.predict_step(u)
        self.theta += self._theta_dot * self.dt   # dead-reckon theta (unobservable, decoupled)
