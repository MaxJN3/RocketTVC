def rk4_step(dynamics, t, x, u, dt):
    k1 = dynamics(t, x, u)
    k2 = dynamics(t + dt/3.0, x + (dt/3.0)*k1, u)
    k3 = dynamics(t + 2.0*dt/3.0, x - (dt/3.0)*k1 + dt*k2, u)
    k4 = dynamics(t + dt, x + dt*k1 - dt*k2 + dt*k3, u)
    
    x_next = x + (dt / 8.0) * (k1 + 3*k2 + 3*k3 + k4)
    return x_next