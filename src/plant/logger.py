import numpy as np
import matplotlib.pyplot as plt

class FlightLogger:
    def __init__(self, state_labels=None, input_labels=None, u_limit_deg=5.0):
        self.u_limit_deg = u_limit_deg
        self.history = {
            "time": [],
            "x": [],
            "u": [],
            "pos": [],
            "vel": []
        }
        
        self.state_labels = state_labels or [
            "Pitch", "Pitch Rate", "Servo Pitch"
        ]
        self.input_labels = input_labels or [
            "Pitch Command"
        ]
        
        # Dynamically determine the number of columns
        self.has_yaw = any("yaw" in lbl.lower() for lbl in self.state_labels)

    def log_step(self, t, x, u, pos=None, vel=None):
        self.history["time"].append(t)
        self.history["x"].append(np.copy(x).flatten())
        
        if u is not None:
            self.history["u"].append(np.copy(u).flatten())
        else:
            self.history["u"].append(np.zeros(len(self.input_labels)))
            
        # Log position and velocity if provided
        if pos is not None and vel is not None:
            self.history["pos"].append(np.copy(pos))
            self.history["vel"].append(np.copy(vel))

    def plot_dashboard(self):
        """Generates the main control dashboard."""
        time = np.array(self.history["time"])
        X = np.array(self.history["x"])
        U = np.array(self.history["u"])
        
        cols = 2 if self.has_yaw else 1
        fig, axs = plt.subplots(3, cols, figsize=(7 * cols, 10), sharex=True)
        fig.suptitle('Rocket TVC Simulation Results', fontsize=16, fontweight='bold')
        
        # Normalize axs to always be 2D so our matrix indexing [row, col] never breaks!
        axs = np.array(axs).reshape(3, cols)

        # --- Route the Data ---
        for i, label in enumerate(self.state_labels):
            if i >= X.shape[1]: break
            lbl_low = label.lower()
            
            col = 1 if "yaw" in lbl_low and self.has_yaw else 0
            if "rate" in lbl_low or "dot" in lbl_low: continue
                
            if "servo" in lbl_low or "gimbal" in lbl_low or "\u03b4" in lbl_low:
                axs[2, col].plot(time, np.degrees(X[:, i]), label=f"{label} (Actual)", color='#9467bd', lw=2)
            else:
                axs[0, col].plot(time, np.degrees(X[:, i]), label=label, color='#1f77b4', lw=2)

        for j, label in enumerate(self.input_labels):
            if j >= U.shape[1]: break
            lbl_low = label.lower()
            col = 1 if "yaw" in lbl_low and self.has_yaw else 0
            axs[1, col].plot(time, np.degrees(U[:, j]), label=f"{label} (Cmd)", color='#d62728', lw=2)

        # --- Shared Formatting ---
        titles = [
            ["Pitch Kinematics", "Yaw Kinematics"],
            ["Pitch Command (\u03b4_cmd)", "Yaw Command (\u03b4_cmd)"],
            ["Pitch Servo (\u03b4)", "Yaw Servo (\u03b4)"]
        ]
        
        for row in range(3):
            for col in range(cols):
                ax = axs[row, col]
                ax.set_title(titles[row][col], loc='left')
                ax.set_ylabel('Degrees')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
                
                if row == 1 or row == 2:
                    ax.axhline(self.u_limit_deg, color='gray', linestyle=':', label='Limit' if col==0 else "")
                    ax.axhline(-self.u_limit_deg, color='gray', linestyle=':')
                    
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc='upper right')
                if row == 2:
                    ax.set_xlabel('Time (Seconds)')

        plt.tight_layout()
        plt.show()

        # If trajectory data was logged, plot the second figure!
        if len(self.history["pos"]) > 0:
            self.plot_trajectory()

    def plot_trajectory(self):
        """Generates a separate plot for flight path and velocity (handles 2D and 3D)."""
        time = np.array(self.history["time"])
        pos = np.array(self.history["pos"])
        vel = np.array(self.history["vel"])
        
        # Check if we are logging 2D or 3D kinematics
        is_3d = pos.shape[1] >= 3
        
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle('Rocket Flight Trajectory', fontsize=16, fontweight='bold')
        
        # --- Plot 1: Flight Path (2D or 3D) ---
        if is_3d:
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='#2ca02c', lw=2)
            ax1.set_title('3D Flight Path')
            ax1.set_xlabel('Downrange X (m)')
            ax1.set_ylabel('Crossrange Y (m)')
            ax1.set_zlabel('Altitude Z (m)')
        else:
            ax1 = fig.add_subplot(121)
            ax1.plot(pos[:, 0], pos[:, 1], color='#2ca02c', lw=2)
            ax1.set_title('2D Flight Path')
            ax1.set_xlabel('Downrange X (m)')
            ax1.set_ylabel('Altitude Y (m)')
            ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax1.axis('equal') 
            
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- Plot 2: Velocities over time ---
        ax2 = fig.add_subplot(122)
        ax2.plot(time, vel[:, 0], label='Vel X (Downrange)', color='#1f77b4', lw=2)
        
        if is_3d:
            ax2.plot(time, vel[:, 1], label='Vel Y (Crossrange)', color='#d62728', lw=2)
            ax2.plot(time, vel[:, 2], label='Vel Z (Vertical)', color='#ff7f0e', lw=2)
        else:
            ax2.plot(time, vel[:, 1], label='Vel Y (Vertical)', color='#ff7f0e', lw=2)
            
        ax2.set_title('Velocity Profiles')
        ax2.set_xlabel('Time (Seconds)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()