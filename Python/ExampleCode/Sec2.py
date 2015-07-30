'''
------------------------------------------------------------------------
Last updated 7/8/2015

This program runs the steady state solver as well as the time path
iteration solution.

This Python script calls the following other file(s) with the associated
functions:
    Sec2ssfuncs.py
        feasible
        get_L
        get_K
        get_Y
        get_C
        get_r
        get_w
        get_cvec_ss
        get_b_errors
        EulerSys
        SS
    Sec2tpfuncs.py
        TPI
------------------------------------------------------------------------
'''
# Import packages

import numpy as np
import Sec2ssfuncs as s2ssf
reload(s2ssf)
import Sec2tpfuncs as s2tpf
reload(s2tpf)

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S            = integer in [3,80], number of periods an individual lives
T            = integer > S, number of time periods until steady state
beta_annual  = scalar in [0,1), discount factor for one year
beta         = scalar in [0,1), discount factor for each model period
sigma        = scalar > 0, coefficient of relative risk aversion
A            = scalar > 0, total factor productivity parameter in firms'
               production function
alpha        = scalar in (0,1), capital share of income
delta_annual = scalar in [0,1], one-year depreciation rate of capital
delta        = scalar in [0,1], model-period depreciation rate of
               capital
SS_tol       = scalar > 0, tolerance level for steady-state fsolve
SS_graphs    = boolean, =True if want graphs of steady-state objects
TPI_solve    = boolean, =True if want to solve TPI after solving SS
TPI_tol      = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI  = integer >= 1, Maximum number of iterations for TPI
mindist_TPI  = scalar > 0, Convergence criterion for TPI
xi           = scalar in (0,1], TPI path updating parameter
TPI_graphs   = boolean, =True if want graphs of TPI objects
------------------------------------------------------------------------
'''
# Household parameters
S = int(80)
T = int(round(2.5 * S))
beta_annual = .96
beta = beta_annual ** (80 / S)
sigma = 3.0
# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (80 / S))
# SS parameters
SS_tol = 1e-13
SS_graphs = True
# TPI parametersjj
TPI_solve = Falsje
TPI_tol = 1e-13
maxiter_TPI = 100
mindist_TPI = 1e-13
xi = .20
TPI_graphs = False

'''
------------------------------------------------------------------------
Compute the steady state
------------------------------------------------------------------------
b_guess       = [S-1,] vector, initial guess for steady-state
                distribution of savings
feas_params   = [4,] vector, parameters for feasible function
                [S, A, alpha, delta]
GoodGuess     = boolean, =True if initial steady-state guess is feasible
K_constr_init = boolean, =True if K<=0 for initial guess b_guess
c_constr_init = [S,] boolean vector, =True if c<=0 for initial b_guess
ss_ints       = [1,] vector, integer parameters to be passed to SS func
ss_params     = [6,] vector, parameters to be passed in to SS function
b_ss          = [S-1,] vector, steady-state distribution of savings
c_ss          = [S,] vector, steady-state distribution of consumption
w_ss          = scalar > 0, steady-state real wage
r_ss          = scalar > 0, steady-state real interest rate
K_ss          = scalar > 0, steady-state aggregate capital stock
EulErr_ss     = [S-1,] vector, steady-state Euler errors
L_ss          = scalar > 0, steady-state aggregate labor
Y_params      = [2,] vector, production function parameters [A, alpha]
Y_ss          = scalar > 0, steady-state aggregate output (GDP)
C_ss          = scalar > 0, steady-state aggregate consumption
rcdiff_ss     = scalar, steady-state difference in goods market clearing
                (resource constraint)
------------------------------------------------------------------------
'''
# Make initial guess of the steady-state
b_guess = 0.01 * np.ones(S-1)
# Make sure initial guess is feasible
feas_params = np.array([S, A, alpha, delta])
GoodGuess, K_constr_init, c_constr_init = s2ssf.feasible(feas_params, b_guess)
if K_constr_init == True and c_constr_init.max() == False:
    print 'Initial guess is not feasible because K<=0. Some element(s) of bvec must increase.'
elif K_constr_init == False and c_constr_init.max() == True:
    print 'Initial guess is not feasible because some element of c<=0. Some element(s) of bvec must decrease.'
elif K_constr_init == True and c_constr_init.max() == True:
    print 'Initial guess is not feasible because K<=0 and some element of c<=0. Some element(s) of bvec must increase and some must decrease.'
elif GoodGuess == True:
    print 'Initial guess is feasible.'

    # Compute steady state
    print 'BEGIN STEADY STATE COMPUTATION'
    ss_ints = np.array([S])
    ss_params = np.array([beta, sigma, A, alpha, delta, SS_tol])
    b_ss, c_ss, w_ss, r_ss, K_ss, EulErr_ss = \
        s2ssf.SS(ss_ints, ss_params, b_guess, SS_graphs)

    # Print diagnostics
    print 'The maximum absolute steady-state Euler error is: ', np.absolute(EulErr_ss).max()
    print 'The steady-state distribution of capital is:'
    print b_ss
    print 'The steady-state distribution of consumption is:'
    print c_ss
    print 'The steady-state wage, interest rate, and aggregate capital are:'
    print np.array([w_ss, r_ss, K_ss])
    L_ss = s2ssf.get_L(np.ones(S))
    Y_params = np.array([A, alpha])
    Y_ss = s2ssf.get_Y(Y_params, K_ss, L_ss)
    C_ss = s2ssf.get_C(c_ss)
    rcdiff_ss = Y_ss - C_ss - delta * K_ss
    print 'The difference Ybar - Cbar - delta * Kbar is: ', rcdiff_ss

    '''
    --------------------------------------------------------------------
    Compute the equilibrium time path by TPI
    --------------------------------------------------------------------
    Gamma1     = [S-1,] vector, initial period savings distribution
    K1         = scalar > 0, initial period aggregate capital stock
    Kpath_init = [T+S-1,] vector, initial guess for the time path of
                 the aggregate capital stock

    [TODO]
    --------------------------------------------------------------------
    '''
    if TPI_solve == True:
        print 'BEGIN EQUILIBRIUM TIME PATH COMPUTATION'
        Gamma1 = 0.9 * b_ss
        # Make sure init. period distribution is feasible in terms of K
        K1, K_constr_tpi1 = s2ssf.get_K(Gamma1)
        if K1 <= 0:
            print 'Initial savings distribution is not feasible because K1<=0. Some element(s) of Gamma1 must increase.'
        else:
            # Choose initial guess of path of aggregate capital stock
            Kpath_init = np.ones(T+S-1)
            Kpath_init[:T] = np.linspace(K1, K_ss, T)
            Kpath_init[T:] = K_ss
            # Generate path of aggregate labor
            Lpath = (np.ones(S)).sum() * np.ones(T+S-1)

            # Run TPI
            tpi_ints = np.array([S, T])
            tpi_params = np.array([beta, sigma, A, alpha, delta,
                         K1, K_ss, maxiter_TPI, mindist_TPI, xi,
                         TPI_tol])
            b_path, c_path, K_path, w_path, r_path, EulErr_path = \
                s2tpf.TPI(tpi_ints, tpi_params, Kpath_init, Gamma1,
                Lpath, TPI_graphs)

            # Testing functions code

            # p = 2
            # rpath = np.array([0.5, 0.6])
            # wpath = np.array([0.10, 0.15])
            # bvec2 = 0.01 * np.ones(p)
            # cvec, c_constr = s2f.get_cvec_lf(p, rpath, wpath, bvec2)
