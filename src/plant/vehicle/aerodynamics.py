import numpy as np

from src.plant.parameters import AirframeParams, AeroParams

class Aerodynamics:
    def __init__(self, airframe: AirframeParams, aero: AeroParams):
        self.radius = airframe.radius
        self.ref_area = np.pi * (self.radius ** 2)

        self.cp_distance = airframe.cp_from_nose

        self.cd = aero.cd
        self.cna = aero.cna

        self.air_density = aero.air_density

    def calculate_forces_and_torque(self, vx, vy, pitch, cg_distance, wind_x=0.0, wind_y=0.0):
        """
        Calculates aerodynamic forces and torques acting on the rocket.
        
        Args:
            vx (float): Global X velocity (downrange) in m/s.
            vy (float): Global Y velocity (altitude) in m/s.
            pitch (float): Rocket pitch angle in radians (0 is straight up, +X is positive).
            cg_distance (float): Distance from the NOSE to the Center of Gravity (CG) in meters.
            wind_x (float): Wind velocity in X direction in m/s.
            wind_y (float): Wind velocity in Y direction in m/s.
            
        Returns:
            dict: Containing global forces (F_aero_x, F_aero_y) and aerodynamic torque (M_aero).
        """
        v_rel_x = vx - wind_x
        v_rel_y = vy - wind_y
        
        V_mag = np.sqrt(v_rel_x**2 + v_rel_y**2)
        
        if V_mag < 0.1:
            return {"F_aero_x": 0.0, "F_aero_y": 0.0, "M_aero": 0.0}

        q = 0.5 * self.air_density * (V_mag ** 2)

        # Flight path angle (gamma) - assuming Y is up, X is right
        gamma = np.arctan2(v_rel_x, v_rel_y) 
        
        # Angle of attack
        alpha = pitch - gamma
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi 

        F_drag = q * self.ref_area * self.cd
        
        F_drag_x = -F_drag * (v_rel_x / V_mag)
        F_drag_y = -F_drag * (v_rel_y / V_mag)
        
        F_normal = q * self.ref_area * self.cna * alpha
        
        F_normal_x = F_normal * np.cos(pitch)
        F_normal_y = -F_normal * np.sin(pitch)
        
        F_aero_x = F_drag_x + F_normal_x
        F_aero_y = F_drag_y + F_normal_y

        
        moment_arm = cg_distance - self.cp_distance
        
        # A positive alpha (wind hitting from the left) creates a normal force pushing right.
        # If CP is above CG, pushing right on the top of the rocket creates a POSITIVE pitch torque (destabilizing).
        M_aero = F_normal * moment_arm

        return {
            "F_aero_x": F_aero_x,
            "F_aero_y": F_aero_y,
            "M_aero": M_aero,
            "alpha": alpha
        }