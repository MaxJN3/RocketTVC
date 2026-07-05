def rk4_step(f, t, x, dt, *args, **kwargs):
    """
    One step of 4th-order Runge-Kutta (3/8 rule): f(t, x, *args, **kwargs) -> dx/dt.

    Any extra positional/keyword arguments are passed through to f unchanged —
    they are inputs held constant over the step (zero-order hold), e.g. the
    control input, aero moment, thrust, or mass.
    """
    k1 = f(t, x, *args, **kwargs)
    k2 = f(t + dt/3.0, x + (dt/3.0)*k1, *args, **kwargs)
    k3 = f(t + 2.0*dt/3.0, x - (dt/3.0)*k1 + dt*k2, *args, **kwargs)
    k4 = f(t + dt, x + dt*k1 - dt*k2 + dt*k3, *args, **kwargs)

    x_next = x + (dt / 8.0) * (k1 + 3*k2 + 3*k3 + k4)
    return x_next
