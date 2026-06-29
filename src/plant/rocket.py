import numpy as np

class Rocket():
    def __init__(self):
        self.dry_mass = 0.85
        self.wet_mass = 1.0
        self.propellant_mass = self.wet_mass - self.dry_mass
        
        #Motor
        self.burn_time = 3.4
        self.average_thrust = 15.0
        
        #Geometry
        self.length = 1.0
        
        #Init
        self.thrust = self.average_thrust
        self.mass = self.wet_mass
        self.inertia = (1.0 / 12.0) * self.mass * (self.length ** 2)
        self.cg_distance = self.length / 2.0
        
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
            
        self.inertia = (1.0 / 12.0) * self.m * (self.length ** 2)
        
        self.cg_distance = self.length / 2
    
    def dynamics(self, t, x, u):
        # x = [theta, theta_dot]
        # u = [delta] (gimbal angle)
        theta = x[0]
        theta_dot = x[1]
        delta = u[0]
        
        m = self.mass(t) 
        I = self.inertia(t)
        T = self.thrust(t)
        l_cg = self.cg_distance(t)
        
        theta_ddot = (-T * np.sin(delta) * l_cg) / I
        
        return np.array([theta_dot, theta_ddot])