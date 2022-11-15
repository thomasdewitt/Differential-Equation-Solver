import numpy as np
from numba import njit



def solve(ic, func, dt, end_time, save_data=False, jit=False, dtype=np.float32):
    """
        Solve coupled differential equations of the form 
            \\frac{d\phi_1}{dt} = f_1(\phi_1, \phi_2,...\phi_n)
            \\frac{d\phi_2}{dt} = f_2(\phi_1, \phi_2,...\phi_n)
            ...
            \\frac{d\phi_n}{dt} = f_n(\phi_1, \phi_2,...\phi_n)

        Inputs:
            ic: np.ndarray 1-D, Initial conditions [phi_1, phi_2, ..., phi_n]    at t=0
            func: function which takes a single arg (\phi_1, \phi_2,...\phi_n) and returns [f_1(\phi_1, \phi_2,...\phi_n), f_2(\phi_1, \phi_2,...\phi_n), ...,f_n(\phi_1, \phi_2,...\phi_n)]
            dt: float, time step
            end_time: float, how long to simulate for

            Options:
            save_data: Whether to keep values [phi_1, phi_2, ..., phi_n] for every time step
            jit: whether func functions have been jitted (with @njit())

        Output:
            if save_data: 2-D np.ndarray where rows are time step, cols are variables [phi_1, phi_2, ..., phi_n] e.g.
                [[t_1, phi_1, phi_2, ..., phi_n]         # at t = 0
                 [t_2, phi_1, phi_2, ..., phi_n]         # at t = 1 dt
                 ...
                 [t_n, phi_1, phi_2, ..., phi_n]]        # t_n = end_time
            else:
                [phi_1, phi_2, ..., phi_n]          # at t = end_time
    """
    if type(func) == type(solve) and jit: raise ValueError("If jit=True 'func' must be jitted (with @njit())")
    ic = np.asarray(ic, dtype=dtype)
    test = func(ic)
    if type(test) != np.ndarray or test.dtype != dtype: raise ValueError('"func" must return a np.ndarray with dtype "dtype"')
    
    if save_data:
        timesteps = np.linspace(0, end_time, int(end_time/dt+1), dtype=dtype)
        vars = np.empty((timesteps.shape[0], len(ic)+1), dtype=dtype)  # cols are timesteps, phi1, ph2, ...
        vars[:,0] = timesteps
        vars[0,1:] = ic
        vars[1,1:] = ic     # set first two timesteps to ic
        if jit: vars = loop_jit_save(vars, func, dt)
        else: vars = loop_save(vars, func, dt)
    else: 
        if jit: vars = loop_jit(ic, ic.copy(), func, dtype(dt), int(end_time/dt))
        else: vars = loop(ic, ic.copy(), func, dtype(dt), int(end_time/dt))

    if vars.max() > 1e20 or (vars.min() < -1e20): 
        raise OverflowError('Solution diverged. Set dt smaller?')
    return vars

@njit()
def loop_jit_save(vars, func, dt):

    for i in range(vars.shape[0]-1):
        increment = func(vars[i,1:])*dt
        vars[i+1,1:] = vars[i,1:] + increment

    return vars

def loop_save(vars, func, dt):
    for i in range(vars.shape[0]-1):
        increment = func(vars[i,1:])*dt
        vars[i+1,1:] = vars[i,1:] + increment

    return vars

@njit()
def loop_jit(ic, ic_plus_1, func, dt, n_timesteps):
    current_vars = ic_plus_1
    last_vars = ic
    
    for _ in range(n_timesteps):
        increment = func(last_vars)*dt
        current_vars = last_vars+increment
        last_vars = current_vars

    return current_vars

def loop(ic, ic_plus_1, func, dt, n_timesteps):
    current_vars = ic_plus_1
    last_vars = ic
    
    for _ in range(n_timesteps):
        increment = func(last_vars)*dt
        current_vars = last_vars+increment
        last_vars = current_vars

    return current_vars
