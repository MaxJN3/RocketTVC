import numpy as np
import matplotlib.pyplot as plt

class FlightLogger:
    def __init__(self):
        self.history = {
            "time": [],
            "theta_deg": [],
            "theta_dot_deg": [],
            "gimbal_deg": [],
            "thrust_N": [],
            "mass_kg": []
        }

    def log_step(self, t, x, u, thrust, mass):
        """Records a single time step of data."""
        self.history["time"].append(t)
        self.history["theta_deg"].append(np.degrees(x[0]))
        self.history["theta_dot_deg"].append(np.degrees(x[1]))
        
        gimbal_angle = np.degrees(u[0]) if u is not None else 0.0
        self.history["gimbal_deg"].append(gimbal_angle)
        
        self.history["thrust_N"].append(thrust)
        self.history["mass_kg"].append(mass)

    def plot_dashboard(self):
        """Generates the Matplotlib dashboard from the logged data."""
        time = self.history["time"]
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle('Rocket TVC Simulation Results', fontsize=16, fontweight='bold')

        # 1. Pitch Angle
        axs[0].plot(time, self.history["theta_deg"], label='Pitch Angle', color='#1f77b4', lw=2)
        axs[0].axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
        axs[0].set_ylabel('Degrees')
        axs[0].set_title('Rocket Pitch Angle', loc='left')
        axs[0].grid(True, linestyle=':', alpha=0.6)
        axs[0].legend(loc='upper right')

        # 2. Gimbal Angle
        axs[1].plot(time, self.history["gimbal_deg"], label='Gimbal Command', color='#d62728', lw=2)
        axs[1].axhline(5, color='gray', linestyle=':', label='Max Deflection')
        axs[1].axhline(-5, color='gray', linestyle=':')
        axs[1].set_ylabel('Degrees')
        axs[1].set_title('TVC Gimbal Actuation', loc='left')
        axs[1].grid(True, linestyle=':', alpha=0.6)
        axs[1].legend(loc='upper right')

        # 3. Hardware State (Dual Axis)
        ax3_mass = axs[2].twinx()
        axs[2].plot(time, self.history["thrust_N"], label='Thrust', color='#ff7f0e', lw=2)
        ax3_mass.plot(time, self.history["mass_kg"], label='Mass', color='#2ca02c', ls='--', lw=2)

        axs[2].set_xlabel('Time (Seconds)')
        axs[2].set_ylabel('Thrust (N)', color='#ff7f0e', fontweight='bold')
        ax3_mass.set_ylabel('Mass (kg)', color='#2ca02c', fontweight='bold')
        axs[2].set_title('Engine Telemetry', loc='left')
        axs[2].grid(True, linestyle=':', alpha=0.6)

        # Merge legends for the dual-axis plot
        lines_1, labels_1 = axs[2].get_legend_handles_labels()
        lines_2, labels_2 = ax3_mass.get_legend_handles_labels()
        axs[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

        plt.tight_layout()
        plt.show()