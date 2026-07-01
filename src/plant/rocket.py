import numpy as np

class Rocket():
    def __init__(self):
        self.dry_mass = 0.85
        self.wet_mass = 1.0
        self.propellant_mass = self.wet_mass - self.dry_mass
        
        #Motor
        self.burn_time = 5
        self.average_thrust = 15.0
        
        #Geometry
        self.length = 1.0
        self.radius = 0.04
        
        self.cg = self.length / 2.0
        self.cp = 0.4
        
        #Init
        self.thrust = self.average_thrust
        self.mass = self.wet_mass
        self.inertia = (1.0 / 12.0) * self.mass * (self.length ** 2)
        
    def update_state(self, t):
        '''
        Thrust in Newtons. 0 if the motor has burned out.
        Mass drops linearly as solid fuel burns
        Moment of inertia for a long thin cylinder: I = (1/12) * m * L^2
        Distance from the Center of Gravity (CG) to the motor gimbal.
        '''
        
        self.thrust = self.average_thrust if t < self.burn_time else 0.0
        
        if t < self.burn_time:
            burn_fraction = t / self.burn_time
            current_mass = self.wet_mass - (self.propellant_mass * burn_fraction)
            self.mass = current_mass
        else:
            self.mass = self.dry_mass
            
        self.inertia = (1.0 / 12.0) * self.mass * (self.length ** 2)
        
        self.cg = self.length / 2
    
    def dynamics(self, t, x, u, M_aero, omega_c=20):
        # x = [theta, theta_dot]
        # u = [delta] (gimbal angle)
        theta, theta_dot, delta = x
        delta_cmd = u[0]
        
        alpha = (-self.thrust * self.cg) / self.inertia
        
        theta_ddot = alpha * delta + M_aero / self.inertia
        
        delta_dot = omega_c * (delta_cmd - delta)
        
        return np.array([theta_dot, theta_ddot, delta_dot])