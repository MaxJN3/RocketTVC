# Rocket Thrust Vector Control (TVC) Simulation

A Python simulation framework for rocket pitch attitude control using Model Predictive Control (MPC) and state estimation under dynamic mass depletion and aerodynamic wind disturbances.

The goal of this project is to port the MPC and Kalman control to a microcontroller in C++ to control the pitch of a 3D printed rocket in real-time.

## System Architecture

The simulation models a 2D rocket pitch-axis system driven by a single gimbaled motor, coupled with aerodynamic drag, normal forces, and IMU sensor dynamics.

* **State-Space Model:** Dynamic 3-state system representing $[\theta, \dot{\theta}, \delta]^T$ (pitch angle, pitch rate, gimbal angle), augmented with an unmeasured disturbance state (crosswind velocity).
* **Online Re-linearization:** System dynamics ($\Phi, \Gamma$) are updated at each time step to account for time-varying mass, center of gravity (CG), and moment of inertia ($I$).
* **Control (MPC):** Formulated via `cvxpy` with hard input/rate bounds, soft state constraints using slack penalty vectors, and terminal cost matrices ($Q_f$).
* **State Estimation (Kalman Filter):** Reduced-order Kalman Filter running on gyro pitch-rate measurements. Solves the Discrete Algebraic Riccati Equation (DARE) with an iterative matrix solver fallback for non-convergent operating points.
* **Physics Engine:** Plant dynamics and vehicle kinematics propagated using 4th-order Runge-Kutta (RK4) integration.

## Project Structure

* `src/controlling/controllers/` - MPC and Discrete Kalman Filter implementations.
* `src/controlling/estimators/` - Sensor fusion and attitude estimation.
* `src/controlling/modelling/` - State-space linearizations and transformation pipelines.
* `src/plant/` - Physical vehicle models, aerodynamic calculations, IMU sensor noise models, and RK4 numerical solvers.

## Roadmap
* **Add Dimension:** Simulate in 3D.
* **Tune Controller:** Tune the MPC and Kalman parameters.
* **C++ Porting:** Port matrix solvers and MPC optimization routines to C++ for microcontroller execution.


## Execution

Run the main TVC simulation and render the flight dashboard:
```bash
python -m src.plant.simulation

