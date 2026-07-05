import numpy as np
from src.plant.diffeqsolvers import rk4_step

class Kinematics:
    def __init__(self, is_3d=False):
        """
        A general kinematic tracking system.
        If is_3d=False: Tracks 2D Space (X = Downrange, Y = Altitude)
        If is_3d=True:  Tracks 3D Space (X = Downrange, Y = Crossrange, Z = Altitude)
        """
        self.is_3d = is_3d
        
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        
        self.state = np.zeros(6 if is_3d else 4)

    def dynamics(self, t, state, thrust, mass, pitch_angle, yaw_angle=0.0, fx_aero=0.0, fy_aero=0.0, fz_aero=0.0):
        """
        Derivative calculation function required by the RK4 solver.
        """
        if self.is_3d:
            vx, vy, vz = state[3], state[4], state[5]
            
            # Thrust in the Body Frame
            # Assuming the rocket's longitudinal axis is Z
            F_body = np.array([0.0, 0.0, thrust])
            
            # Define Rotation Matrices
            # Pitch (Rotation about Y-axis) -> Tilts nose into +X (Downrange)
            R_pitch = np.array([
                [ np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                [ 0,                   1, 0                  ],
                [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
            ])
            
            # Yaw (Rotation about X-axis) -> Tilts nose into +Y (Crossrange)
            R_yaw = np.array([
                [ 1, 0,                    0                   ],
                [ 0, np.cos(yaw_angle),   -np.sin(yaw_angle)   ],
                [ 0, np.sin(yaw_angle),    np.cos(yaw_angle)   ]
            ])
            
            R_global = R_yaw @ R_pitch 
            F_inertial = R_global @ F_body
            
            # (a = F/m)
            ax = (F_inertial[0] + fx_aero) / mass
            ay = (F_inertial[1] + fy_aero) / mass
            az = ((F_inertial[2] + fz_aero) / mass) - 9.81
            
            return np.array([vx, vy, vz, ax, ay, az])
            
        else:
            # 2D Case (X = Downrange, Y = Altitude)
            vx, vy = state[2], state[3]
            
            ax = (thrust / mass) * np.sin(pitch_angle) + (fx_aero / mass)
            ay = (thrust / mass) * np.cos(pitch_angle) - 9.81 + (fy_aero / mass)
            
            return np.array([vx, vy, ax, ay])

    def step(self, dt, thrust, mass, pitch_angle, yaw_angle=0.0, fx_aero=0.0, fy_aero=0.0, fz_aero=0.0):
        """Advances the spatial physics forward by dt using numerical RK4 integration."""
        self.state = rk4_step(self.dynamics, 0.0, self.state, dt,
                              thrust=thrust, mass=mass, pitch_angle=pitch_angle, yaw_angle=yaw_angle,
                              fx_aero=fx_aero, fy_aero=fy_aero, fz_aero=fz_aero)
        
        if self.is_3d:
            self.x, self.y, self.z, self.vx, self.vy, self.vz = self.state
        else:
            self.x, self.y, self.vx, self.vy = self.state

    @property
    def pos(self):
        """Returns position array correctly formatted for the FlightLogger."""
        return self.state[0:3] if self.is_3d else self.state[0:2]

    @property
    def vel(self):
        """Returns velocity array correctly formatted for the FlightLogger."""
        return self.state[3:6] if self.is_3d else self.state[2:4]