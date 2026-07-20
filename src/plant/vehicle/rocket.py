import numpy as np

from src.plant.parameters import VehicleParams


class Rocket():
    """Physical model: mass properties, thrust, and rotational dynamics."""

    def __init__(self, params: VehicleParams):
        self.params = params

        self.length = params.airframe.length
        self.radius = params.airframe.radius
        self.dry_mass = params.airframe.dry_mass
        self.wet_mass = self.dry_mass + params.motor.propellant_mass

        self.update_state(0.0)

    @property
    def gimbal_arm(self):
        """Distance from the CG to the motor gimbal at the tail"""
        return self.length - self.cg

    def update_state(self, t):
        '''Thrust in Newtons. 0 if the motor has burned out'''
        self.thrust = self.params.motor.thrust(t)
        self.mass = self.dry_mass + self.params.motor.propellant_remaining(t)
        self.inertia = (1.0 / 12.0) * self.mass * (self.length ** 2)

        # Constant for now
        self.cg = self.length / 2

    def dynamics(self, t, x, u, M_aero):
        # x = [theta, theta_dot, delta]
        # u = [delta_cmd] (commanded gimbal angle)
        theta, theta_dot, delta = x
        delta_cmd = u[0]

        alpha = (-self.thrust * self.gimbal_arm) / self.inertia

        theta_ddot = alpha * delta + M_aero / self.inertia

        delta_dot = self.params.actuator.omega_c * (delta_cmd - delta)

        return np.array([theta_dot, theta_ddot, delta_dot])
