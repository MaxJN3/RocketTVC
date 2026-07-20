# 2D Rocket Thrust Vector Control Simulation
A Guidance, Navigation, and Control simulation evaluating a state-space Model Predictive Controller coupled with an Adaptive Kalman Filter for attitude estimation under dynamic atmospheric disturbances.

## Implementations

*   **Dynamic State-Space Linearization:** The plant physics change dynamically as mass depletes and thrust curves shift. The simulation continuously re-linearizes the system matrices ($\Phi, \Gamma$) at each time step around the current operating point.

*   **Disturbance-Observer MPC:** Implements an MPC framework mapping $[\theta, \dot{\theta}, \delta]^T$. The state-space model is extended with an unmeasured disturbance state to estimate and reject crosswinds.

*   **IMU:** Implements an IMU gyroscope sensor model with stochastic noise.

*   **State Estimation:** A reduced-order Kalman Filter estimates the observable subsystem in real-time, providing the MPC with filtered state inputs ($\hat{x}$) under measurement uncertainty.

*   **Numerical Integration:** Plant propagation is driven by a 4th-order Runge-Kutta, preserving physics during high-frequency actuator dynamics.

