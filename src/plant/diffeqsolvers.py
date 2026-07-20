def rk4_step(f, t, x, dt, *args, **kwargs):
    k1 = f(t, x, *args, **kwargs)
    k2 = f(t + dt/3.0, x + (dt/3.0)*k1, *args, **kwargs)
    k3 = f(t + 2.0*dt/3.0, x - (dt/3.0)*k1 + dt*k2, *args, **kwargs)
    k4 = f(t + dt, x + dt*k1 - dt*k2 + dt*k3, *args, **kwargs)

    x_next = x + (dt / 8.0) * (k1 + 3*k2 + 3*k3 + k4)
    return x_next
