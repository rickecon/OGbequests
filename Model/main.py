'''
------------------------------------------------------------------------
Last updated 9/29/2015

This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents and
exogenous labor and intentional bequests.

This Python script calls the following other file(s) with the associated
functions:
    ssfuncs.py
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
    tpfuncs.py
        TPI
------------------------------------------------------------------------
'''
# Import packages

import numpy as np
import ssfuncs as ssf
reload(ssf)
import tpfuncs as tpf
reload(tpf)

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S            = integer in [3,80], number of periods an individual lives
T            = integer > S, number of time periods until steady state
beta_annual  = scalar in [0,1), discount factor for one year
beta         = scalar in [0,1), discount factor for each model period
sigma        = scalar > 0, coefficient of relative risk aversion
chi_b        = scalar > 0, scale parameter on utility of bequests
nvec         = [S,] vector, exogenous labor supply n_{s,t}
L            = scalar > 0, exogenous aggregate labor
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
T = int(round(2 * S)) # Note: T must increase for smaller values of S
beta_annual = .96
beta = beta_annual ** (80 / S)
sigma = 3.0
chi_b = 1
nvec = np.zeros(S)
nvec[:int(round(2 * S / 3))] = 1
nvec[int(round(2 * S / 3)):] = 0.1
L = ssf.get_L(nvec)
# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (80 / S))
# SS parameters
SS_tol = 1e-13
SS_graphs = False
# TPI parameters
TPI_solve = True
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-13
xi = 0.40
TPI_graphs = True

'''
------------------------------------------------------------------------
Compute the steady state
------------------------------------------------------------------------
b_guess          = [S,] vector, initial guess for steady-state
                   distribution of savings
feas_params      = tuple of length 6, parameters for feasible function
                   [S, A, alpha, delta, L, chi_b]
GoodGuess        = boolean, =True if initial steady-state guess is
                   feasible
K_constr_init    = boolean, =True if K<=0 for initial guess b_guess
bsp1_constr_init = boolean, =True if BQ<=0 for initial guess b_guess
c_constr_init    = [S,] boolean vector, =True if c<=0 for initial
                   b_guess
ss_params        = length 9 tuple, parameters to be passed in to SS
                   function:
                   [S, beta, sigma, chi_b, A, alpha, delta, L, SS_tol]
b_ss             = [S-1,] vector, steady-state distribution of savings
c_ss             = [S,] vector, steady-state distribution of consumption
w_ss             = scalar > 0, steady-state real wage
r_ss             = scalar > 0, steady-state real interest rate
BQ_ss            = scalar > 0, steady-state total bequests
K_ss             = scalar > 0, steady-state aggregate capital stock
Y_ss             = scalar > 0, steady-state aggregate output (GDP)
C_ss             = scalar > 0, steady-state aggregate consumption
EulErr_ss        = [S-1,] vector, steady-state Euler errors
ss_time          = scalar, number of seconds to compute SS solution
rcdiff_ss        = scalar, steady-state difference in goods market
                   clearing (resource constraint)
------------------------------------------------------------------------
'''
# Make initial guess of the steady-state
b_guess = np.zeros(S)
b_guess[:int(round(2 * S / 3))] = (np.linspace(0.003, 0.3,
                                   int(round(2 * S / 3))))
b_guess[int(round(2 * S / 3)):] = (np.linspace(0.3, 0.1,
                                   S - int(round(2 * S / 3))))
# Make sure initial guess is feasible
feas_params = (S, A, alpha, delta, L, chi_b)
GoodGuess, K_constr_init, bsp1_constr_init, c_constr_init = \
    ssf.feasible(feas_params, b_guess, nvec)
if K_constr_init == True and bsp1_constr_init == True:
    print 'Initial guess is not feasible because K<=0 and b_{S+1}<=0. Some element(s) of bvec must increase.'
if K_constr_init == True and bsp1_constr_init == False:
    print 'Initial guess is not feasible because K<=0. Some element(s) of bvec must increase.'
elif (K_constr_init == False and bsp1_constr_init == True and
  c_constr_init.max() == True):
    print 'Initial guess is not feasible because b_{S+1}<=0 and some element of c<=0. Some element(s) of bvec must decrease.'
elif (K_constr_init == False and bsp1_constr_init == False and
  c_constr_init.max() == True):
    print 'Initial guess is not feasible because some element of c<=0. Some element(s) of bvec must increase and some must decrease.'
elif (K_constr_init == False and bsp1_constr_init == True and
  c_constr_init.max() == False):
    print 'Initial guess is not feasible because b_{S+1}<=0. b_{S+1} must increase.'
elif GoodGuess == True:
    print 'Initial guess is feasible.'

    # Compute steady state
    print 'BEGIN STEADY STATE COMPUTATION'
    ss_params = (S, beta, sigma, chi_b, A, alpha, delta, L, SS_tol)
    b_ss, c_ss, w_ss, r_ss, BQ_ss, K_ss, Y_ss, C_ss, EulErr_ss, \
        ss_time = ssf.SS(ss_params, b_guess, nvec, SS_graphs)

    # Print diagnostics
    print 'The maximum absolute steady-state Euler error is: ', np.absolute(EulErr_ss).max()
    print 'The steady-state distribution of capital is:'
    print b_ss
    print 'The steady-state distribution of consumption is:'
    print c_ss
    print 'The steady-state wage and interest rate are:'
    print np.array([w_ss, r_ss])
    print 'Aggregate output, capital stock, consumption, and total bequests are:'
    print np.array([Y_ss, K_ss, C_ss, BQ_ss])
    rcdiff_ss = Y_ss - C_ss - delta * K_ss
    print 'The difference Ybar - Cbar - delta * Kbar is: ', rcdiff_ss

    # Print SS computation time
    if ss_time < 60: # seconds
        secs = round(ss_time, 3)
        print 'SS computation time: ', secs, ' sec'
    elif ss_time >= 60 and ss_time < 3600: # minutes
        mins = int(ss_time / 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', mins, ' min, ', secs, ' sec'
    elif ss_time >= 3600 and ss_time < 86400: # hours
        hrs = int(ss_time / 3600)
        mins = int(((ss_time / 3600) - hrs) * 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'
    elif ss_time >= 86400: # days
        days = int(ss_time / 86400)
        hrs = int(((ss_time / 86400) - days) * 24)
        mins = int(((ss_time / 3600) - hrs) * 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', days, ' days,', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'


    '''
    --------------------------------------------------------------------
    Compute the equilibrium time path by TPI
    --------------------------------------------------------------------
    Gamma1        = [S,] vector, initial period savings distribution
    K1            = scalar > 0, initial period aggregate capital stock
    K_constr_tpi1 = boolean, =True if K1<=0 for given Gamma1
    rparams       = length 3 tuple, parameters for r1-function
                    [A, alpha, delta]
    r1            = scalar > 0, initial interest rate based on Gamma1
    BQ1           = scalar > 0, initial total bequests
    Kpath_init    = [T+S-2,] vector, initial guess for the time path of
                    the aggregate capital stock
    aa_K          = scalar, parabola coefficient for Kpath_init
                    Kpath_init = aa*t^2 + bb*t + cc for 0<=t<=T-1
    bb_K          = scalar, parabola coefficient for Kpath_init
    cc_K          = scalar, parabola coefficient for Kpath_init
    BQpath_init   = [T+S-2,] vector, initial guess for the time path of
                    total bequests
    aa_BQ         = scalar, parabola coefficient for BQpath_init
                    BQpath_init = aa*t^2 + bb*t + cc for 0<=t<=T-1
    bb_BQ         = scalar, parabola coefficient for BQpath_init
    cc_BQ         = scalar, parabola coefficient for BQpath_init
    Lpath         = [T+S-2,] vector, exogenous time path for aggregate
                    labor
    tpi_params    = length 17 tuple, (S, T, beta, sigma, chi_b, L, A,
                    alpha, delta, K1, K_ss, BQ_ss, C_ss, maxiter_TPI,
                    mindist_TPI, xi, TPI_tol)
    b_path        = [S, T+S-1] matrix, equilibrium time path of the
                    distribution of savings. Period 1 is the initial
                    exogenous distribution
    c_path        = [S, T+S-2] matrix, equilibrium time path of the
                    distribution of consumption.
    BQ_path       = [T+S-2,] vector, equilibrium time path of total
                    bequests
    w_path        = [T+S-2,] vector, equilibrium time path of the wage
    r_path        = [T+S-2,] vector, equilibrium time path of the
                    interest rate
    K_path        = [T+S-2,] vector, equilibrium time path of the
                    aggregate capital stock
    Y_path        = [T+S-2,] vector, equilibrium time path of aggregate
                    output (GDP)
    C_path        = [T+S-2,] vector, equilibrium time path of aggregate
                    consumption
    EulErr_path   = [S, T+S-1] matrix, equilibrium time path of the
                    Euler errors for all the savings decisions
    tpi_time      = scalar, number of seconds to compute TPI solution
    ResDiff       = [T-1,] vector, errors in the resource constraint
                    from period 1 to T-1. We don't use T because we are
                    missing one individual's consumption in that period
    --------------------------------------------------------------------
    '''
    if TPI_solve == True:
        print 'BEGIN EQUILIBRIUM TIME PATH COMPUTATION'
        Gamma1 = 0.9 * b_ss
        # Make sure init. period distribution is feasible in terms of K
        K1, K_constr_tpi1 = ssf.get_K(Gamma1)
        rparams = (A, alpha, delta)
        r1 = ssf.get_r(rparams, K1, L)
        BQ1 = (1 + r1) * Gamma1[-1]
        if K1 <= 0 and Gamma1[-1] > 0:
            print 'Initial savings distribution is not feasible because K1<=0. Some element(s) of Gamma1 must increase.'
        elif K1 <= 0 and Gamma1[-1] <= 0:
            print 'Initial savings distribution is not feasible because K1<=0 and b_{S+1,1}<=0. Some element(s) of Gamma1 must increase.'
        elif K1 > 0 and Gamma1[-1] <= 0:
            print 'Initial savings distribution is not feasible because b_{S+1,1}<=0. b_{S+1,1} must increase.'
        else:
            # Choose initial guess of path of aggregate capital stock
            # and total bequests. Use parabola specification
            # aa*x^2 + bb*x + cc

            # Initial aggregate capital path
            Kpath_init = np.zeros(T+S-2)
            # Kpath_init[:T] = np.linspace(K1, K_ss, T)
            cc_K = K1
            bb_K = - 2 * (K1 - K_ss) / (T - 1)
            aa_K = -bb_K / (2 * (T - 1))
            Kpath_init[:T] = (aa_K * (np.arange(0, T) ** 2) +
                             (bb_K * np.arange(0, T)) + cc_K)
            Kpath_init[T:] = K_ss

            # Initial total bequests path
            BQpath_init = np.zeros(T+S-2)
            # BQpath_init[:T] = np.linspace(K1, BQ_ss, T)
            cc_BQ = BQ1
            bb_BQ = - 2 * (BQ1 - BQ_ss) / (T - 1)
            aa_BQ = -bb_BQ / (2 * (T - 1))
            BQpath_init[:T] = (aa_BQ * (np.arange(0, T) ** 2) +
                             (bb_BQ * np.arange(0, T)) + cc_BQ)
            BQpath_init[T:] = BQ_ss

            # Generate path of aggregate labor
            Lpath = L * np.ones(T+S-2)

            # Run TPI
            tpi_params = (S, T, beta, sigma, chi_b, L, A, alpha, delta,
                         K1, K_ss, BQ_ss, C_ss, maxiter_TPI,
                         mindist_TPI, xi, TPI_tol)
            (b_path, c_path, BQ_path, w_path, r_path, K_path, Y_path,
                C_path, EulErr_path, tpi_time) = tpf.TPI(tpi_params,
                Kpath_init, BQpath_init, Gamma1, nvec, Lpath, b_ss,
                TPI_graphs)

            # Print diagnostics
            print 'The max. absolute difference in Yt-Ct-K{t+1}+(1-delta)*Kt is:'
            ResDiff = (Y_path[:T-1] - C_path[:T-1] - K_path[1:T] +
                      (1 - delta) * K_path[:T-1])
            print np.absolute(ResDiff).max()

            # Print TPI computation time
            if tpi_time < 60: # seconds
                secs = round(tpi_time, 3)
                print 'TPI computation time: ', secs, ' sec'
            elif tpi_time >= 60 and tpi_time < 3600: # minutes
                mins = int(tpi_time / 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', mins, ' min, ', secs, ' sec'
            elif tpi_time >= 3600 and tpi_time < 86400: # hours
                hrs = int(tpi_time / 3600)
                mins = int(((tpi_time / 3600) - hrs) * 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'
            elif tpi_time >= 86400: # days
                days = int(tpi_time / 86400)
                hrs = int(((tpi_time / 86400) - days) * 24)
                mins = int(((tpi_time / 3600) - hrs) * 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', days, ' days,', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'
