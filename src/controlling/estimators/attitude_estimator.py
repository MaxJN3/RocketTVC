import numpy as np

from src.controlling.controllers.kalman import KalmanFilter


class AttitudeEstimator:
    """
    Rate-gyro attitude estimator (Option 1) for a single pitch axis.

    Three pieces, each placed where the physics makes it observable:
      - gyro bias: calibrated on the pad, where the rocket is clamped and the
        true rate is zero, so the gyro reads bias + noise. The optimal estimate
        of a constant under white noise is the sample mean.
      - theta: dead-reckoned by integrating the filtered rate. It is unobservable
        from a rate gyro and fully decoupled from the dynamics, so integration
        from the known launch angle is all there is — and all that's needed.
      - [theta_dot, delta, wind]: the observable subsystem, handed to the
        steady-state Kalman filter (its DARE solves for this reduced model).

    Produces the full [theta, theta_dot, delta, wind] estimate the MPC expects.
    """

    def __init__(self, kf_model, dt, launch_angle=0.0):
        self.dt = dt
        self.kf = KalmanFilter(kf_model)         # state [theta_dot, delta, wind]
        self.kf.x = np.zeros((kf_model.nbr_states, 1))

        self.theta = launch_angle
        self.gyro_bias = 0.0

        self._theta_dot = 0.0

    def calibrate_bias(self, gyro_samples):
        """Pad phase: average the clamped-rocket gyro readings (true rate = 0)."""
        self.gyro_bias = float(np.mean(gyro_samples))
        return self.gyro_bias

    def update(self, gyro):
        """Correct with the current gyro reading; return full [theta, theta_dot, delta, wind]."""
        omega = gyro - self.gyro_bias

        x_red = self.kf.filter_step(np.array([[omega]]))
        self._theta_dot = float(x_red[0, 0])
        delta = float(x_red[1, 0])
        wind = float(x_red[2, 0])

        return np.array([[self.theta], [self._theta_dot], [delta], [wind]])

    def predict(self, u, Phi_reduced, Gamma_reduced):
        """Propagate to the next step with the freshly computed control u."""
        self.kf.Phi = Phi_reduced
        self.kf.Gamma = Gamma_reduced
        self.kf.predict_step(u)
        self.theta += self._theta_dot * self.dt   # dead-reckon theta (unobservable, decoupled)
