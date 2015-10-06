'''
------------------------------------------------------------------------
Last updated 9/29/2015

All the functions for the SS computation from Chapter 8 of the OG
textbook\
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
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def feasible(params, bvec, nvec):
    '''
    Determines whether a particular guess for the steady-state
    distribution of savings is feasible, first in terms of K>0, then in
    terms of c_s>0 for all s

    Inputs:
        params = length 6 tuple, [S, A, alpha, delta, L, chi_b]
        S      = integer in [3,80], number of periods in a life
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital
        L      = scalar > 0, exogenous aggregate labor
        chi_b  = scalar > 0, scale parameter on utility of bequests
        bvec   = [S,] vector, initial guess for distribution of
                 savings b_{s+1}
        nvec   = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        get_K       = generates aggregate capital stock from bvec
        get_r       = generates interest rate from r_params, K, and L
        get_w       = generates real wage from w_params, K, and L
        get_cvec_ss = generates consumption vector and c_constr from r,
                      w, and bvec

    Objects in function:
        GoodGuess   = boolean, =True if initial steady-state guess is
                      feasible
        bsp1_constr = boolean, =True if b_{S+1}<=0
        K           = scalar > 0, aggregate capital stock
        K_constr    = boolean, =True if K<=0 for given bvec
        c_constr    = [S,] boolean vector, =True if c<=0 for given bvec
        L           = scalar>0, aggregate labor
        r_params    = length 3 tuple, parameters for r-function
                      [A, alpha, delta]
        r           = scalar > 0, interest rate (real return on savings)
        w_params    = length 2 tuple, parameters for w-function
                      [A, alpha]
        w           = scalar > 0, real wage
        BQ          = scalar > 0, total bequests
        cvec        = [S,] vector, consumption c_s for each age-s agent

    Returns: GoodGuess, K_constr, bsp1_constr, c_constr
    '''
    S, A, alpha, delta, L, chi_b = params
    GoodGuess = True
    # Check b_{S+1}
    bsp1_constr = False
    if bvec[-1] <= 0:
        bsp1_constr = True
        GoodGuess = False
    # Check K
    K, K_constr = get_K(bvec)
    if K_constr == True:
        GoodGuess = False
    # Check cvec if K has no problems
    c_constr = np.zeros(S, dtype=bool)
    if K_constr == False and bsp1_constr == False:
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        BQ = get_BQ(r, bvec[-1])
        cvec, c_constr = get_cvec_ss(S, r, w, BQ, bvec, nvec)
        if c_constr.max() == True:
            GoodGuess = False
    return GoodGuess, K_constr, bsp1_constr, c_constr


def get_L(nvec):
    '''
    Generates aggregate labor L from distribution of individual labor
    supply

    Inputs:
        nvec = [S,] vector, distribution of labor supply n_{s}

    Functions called: None

    Objects in function:
        L = scalar, aggregate labor

    Returns: L
    '''
    L = nvec.sum()
    return L


def get_K(barr):
    '''
    Generates aggregate capital stock K from distribution of individual
    savings for either the steady-state or for an entire time path

    Inputs:
        barr = [S,] vector or [S, T] matrix, distribution of savings
               b_{s+1} in steady state or time path for the distribution
               of savings

    Functions called: None

    Objects in function:
        K_constr = boolean or [T,] boolean vector, =True if K<=0 for
                   given bvec in particular period
        K        = scalar or [T,] vector, aggregate capital stock or
                   time path of the aggregate capital stock

    Returns: K, K_constr
    '''
    if barr.ndim == 1:
        # This is the steady-state case
        K = barr.sum()
        K_constr = False
        if K <= 0:
            print 'b vector and/or parameters resulted in K<=0'
            K_constr = True
    elif barr.ndim == 2:
        # This is the time path case
        K = barr.sum(axis=0)
        K_constr = np.zeros(K.shape, dtype=bool)
        if K.min() <= 0:
            print 'Aggregate capital constraint is violated K<=0 for some period in time path.'
            K_constr = K <= 0
    return K, K_constr


def get_Y(params, K, L):
    '''
    Generates aggregate output Y

    Inputs:
        params = length 2 tuple, production function parameters
                 [A, alpha]
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        K      = scalar > 0 or [T+S-1,] vector, aggregate capital stock
                 or time path of the aggregate capital stock
        L      = scalar > 0 or [T+S-1,] vector, aggregate labor or time
                 path of the aggregate labor

    Functions called: None

    Objects in function:
        Y = scalar > 0 or [T+S-1,] vector, aggregate output (GDP) or
            time path of aggregate output (GDP)

    Returns: Y
    '''
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y


def get_C(carr):
    '''
    Generates aggregate consumption C

    Inputs:
        carr = [S,] vector or [S, T-1] matrix, distribution of consumption
               c_s in steady state or time path for the distribution of
               consumption

    Functions called: None

    Objects in function:
        C = scalar > 0 or [T-1, ] vector, aggregate consumption or time
            path of aggregate consumption

    Returns: C
    '''
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)
    return C


def get_r(params, K, L):
    '''
    Generates real interest rate r from parameters, aggregate capital
    stock K, and aggregate labor L

    Inputs:
        params = length 3 tuple, [A, alpha, delta]
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital
        K      = scalar > 0 or [T+S-1,] vector, aggregate capital stock
                 or time path of the aggregate capital stock
        L      = scalar > 0 or [T+S-1,] vector, aggregate labor or time
                 path of the aggregate labor

    Functions called: None

    Objects in function:
        r = scalar > 0 or [T+S-1,] vector, real interest rate (return on
            savings) or time path of interest rate

    Returns: r
    '''
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r


def get_w(params, K, L):
    '''
    Generates real wage w from parameters, aggregate capital stock K,
    and aggregate labor L

    Inputs:
        params = length 2 tuple, [A, alpha]
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        K      = scalar > 0 or [T+S-1,] vector, aggregate capital stock
                 or time path of the aggregate capital stock
        L      = scalar > 0 or [T+S-1,] vector, aggregate labor or time
                 path of the aggregate labor

    Functions called: None

    Objects in function:
        w = scalar > 0 or [T+S-2,] vector, real wage or time path of
            real wage

    Returns: w
    '''
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)
    return w

def get_BQ(r, bq_last):
    '''
    Generates total bequests available BQ in the current period,
    steady-state, or for an entire time path

    Inputs:
        r       = scalar > 0 or [T,] vector, interest rate (real return
                  on savings) or time path of interest rates
        bq_last = scalar > 0, or [T,] vector, last period intended
                  bequests or time path of intended bequests

    Functions called: None

    Objects in function:
        BQ = scalar > 0 or [T,] vector, total bequests available in the
             current period or time path of intended bequests

    Returns: w
    '''
    BQ = (1 + r) * bq_last
    return BQ


def get_cvec_ss(S, r, w, BQ, bvec, nvec):
    '''
    Generates vector of consumptions from distribution of individual
    savings and the interest rate and the real wage

    Inputs:
        S    = integer in [3,80], number of periods an individual lives
        r    = scalar > 0, interest rate
        w    = scalar > 0, real wage
        BQ   = scalar > 0, total bequests available
        bvec = [S,] vector, distribution of savings b_{s+1}
        nvec = [S,] vector, exogenous labor supply n_{s}

    Functions called: None

    Objects in function:
        c_constr = [S,] boolean vector, =True if element c_s <= 0
        b_s      = [S,] vector, 0 in first element and the first S-1
                   elements of bvec in last S-1 elements
        b_sp1    = [S,] vector, bvec
        cvec     = [S,] vector, consumption by age c_s

    Returns: cvec, c_constr
    '''
    c_constr = np.zeros(S, dtype=bool)
    b_s = np.append([0], bvec[:-1])
    b_sp1 = bvec
    cvec = (1 + r) * b_s + w * nvec + (BQ / S) - b_sp1
    if cvec.min() <= 0:
        print 'distribution of savings and/or parameters created c<=0 for some agent(s)'
        c_constr = cvec <= 0
    return cvec, c_constr


def get_b_errors(params, r, cvec, c_constr, bsp1_constr, diff):
    '''
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings

    Inputs:
        params      = length 4 tuple, (beta, sigma, chi_b, bsp1)
        beta        = scalar in [0,1), discount factor
        sigma       = scalar > 0, coefficient of relative risk aversion
        chi_b       = scalar > 0, scale parameter on utility of
                      bequests
        bsp1        = scalar, last period savings (intentional bequests)
        r           = scalar > 0 or [p-1,] vector, interest rate or time
                      path of interest rates with the last value being 0
        cvec        = [p,] vector, distribution of consumption by age
                      c_p
        c_constr    = [p,] boolean vector, =True if c<=0 for given bvec
        bsp1_constr = [S,] boolean vector, last element =True if
                      b_{S+1}<=0
        diff        = boolean, =True if use simple difference Euler
                      errors. Use percent difference errors otherwise.

    Functions called: None

    Objects in function:
        mu_c         = [p-1,] vector, marginal utility of current
                       consumption
        mu_cp1       = [p-1,] vector, marginal utility of next period
                       consumption
        b_errors_dyn = [p-1,] vector, dynamic Euler errors
        b_errors_sta = scalar, static Euler error on intentional
                       bequests
        b_errors     = [p,] vector, Euler errors with errors = 0
                       characterizing optimal savings bvec

    Returns: b_errors
    '''
    beta, sigma, chi_b, bsp1 = params
    cvec[c_constr] = 9999. # Each consumption must be positive to
                           # generate marginal utilities
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    if diff == True:
        b_errors_dyn = (beta * (1 + r) * mu_cp1) - mu_c
        # b_errors_sta = (chi_b ** (-1 / sigma)) * bsp1 - cvec[-1]
        b_errors_sta = chi_b * (bsp1 ** (-sigma)) - mu_cp1[-1]
        b_errors = np.append(b_errors_dyn, b_errors_sta)
        b_errors[np.append(c_constr[1:], False)] = 9999.
        b_errors[c_constr] = 9999.
        b_errors[bsp1_constr] = 9999.
        b_errors[np.append(bsp1_constr[1:], False)] = 9999.
    else:
        b_errors_dyn = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        # b_errors_sta = (chi_b ** (-1 / sigma)) * bsp1 / cvec[-1] - 1
        b_errors_sta = (chi_b * (bsp1 ** (-sigma))) / mu_cp1[-1] - 1
        b_errors = np.append(b_errors_dyn, b_errors_sta)
        b_errors[np.append(False, c_constr[:-1])] = 9999. / 100
        b_errors[c_constr] = 9999. / 100
        b_errors[bsp1_constr] = 9999. / 100
        b_errors[np.append(bsp1_constr[1:], False)] = 9999. / 100
    return b_errors


def EulerSys(bvec, *objs):
    '''
    Generates vector of all Euler errors that characterize all
    optimal lifetime decisions

    Inputs:
        bvec   = [S,] vector, distribution of savings b_{s+1}
        objs   = length 9 tuple,
                 (S, beta, sigma, chi_b, A, alpha, delta, L, nvec)
        S      = integer in [3,80], number of periods an individual
                 lives
        beta   = scalar in [0,1), discount factor
        sigma  = scalar > 0, coefficient of relative risk aversion
        chi_b  = scalar > 0, scale parameter on utility of bequests
        A      = scalar > 0, total factor productivity parameter in firms'
                 production function
        alpha  = scalar in (0,1), capital share of income
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital
        L      = scalar > 0, exogenous aggregate labor
        nvec   = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        get_K        = generates aggregate capital stock from bvec
        get_r        = generates interest rate from r_params, K, and L
        get_w        = generates real wage from w_params, K, and L
        get_cvec_ss  = generates consumption vector and c_constr from
                       r, w, and bvec
        get_b_errors = generates vector of dynamic Euler errors that
                       characterize lifetime savings decisions

    Objects in function:
        K            = scalar > 0, aggregate capital stock
        K_constr     = boolean, =True if K<=0 for given bvec
        b_err_vec    = [S,] vector, vector of Euler errors
        r_params     = length 3 tuple, parameters for r-function
                       [A, alpha, delta]
        r            = scalar > 0, interest rate (real return on
                       savings)
        w_params     = length 2 tuple, parameters for w-function
                       [A, alpha]
        w            = scalar > 0, real wage
        BQ           = scalar > 0, total bequests
        cvec         = [S,] vector, consumption c_s for each age-s
                       agent
        c_constr     = [S,] boolean vector, =True if c<=0 for given
                       bvec
        bsp1_constr  = [S,] boolean vector, last element =True if
                       b_{S+1}<=0
        b_err_params = length 3 tuple, parameters for Euler errors
                       [beta, sigma, chi_b]

    Returns: b_errors
    '''
    S, beta, sigma, chi_b, A, alpha, delta, L, nvec = objs
    K, K_constr = get_K(bvec)
    if K_constr == True:
        b_err_vec = 1000 * np.ones(S)
    else:
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        BQ = get_BQ(r, bvec[-1])
        cvec, c_constr = get_cvec_ss(S, r, w, BQ, bvec, nvec)
        bsp1_constr = np.zeros(S, dtype=bool)
        if bvec[-1] <= 0:
            bsp1_constr[-1] = True
        b_err_params = (beta, sigma, chi_b, bvec[-1])
        b_err_vec = get_b_errors(b_err_params, r, cvec, c_constr,
                                 bsp1_constr, diff=True)
    return b_err_vec


def SS(params, b_guess, nvec, graphs):
    '''
    Generates all endogenous steady-state objects

    Inputs:
        params  = length 9 tuple,
                  [S, beta, sigma, chi_b, A, alpha, delta, L, SS_tol]
        S       = integer in [3,80], number of periods an individual
                  lives
        beta    = scalar in [0,1), discount factor
        sigma   = scalar > 0, coefficient of relative risk aversion
        chi_b   = scalar > 0, scale parameter on utility of bequests
        A       = scalar > 0, total factor productivity parameter in
                  firms' production function
        alpha   = scalar in (0,1), capital share of income
        delta   = scalar in [0,1], model-period depreciation rate of
                  capital
        L       = scalar > 0, exogenous aggregate labor
        SS_tol  = scalar > 0, tolerance level for steady-state fsolve
        b_guess = [S,] vector, initial guess for the distribution
                  of savings b_{s+1}
        nvec    = [S,] vector, exogenous labor supply n_{s}
        graphs  = boolean, =True if want graphs of steady-state objects

    Functions called:
        get_L        = generates aggregate labor from nvec
        get_K        = generates aggregate capital stock from bvec
        get_r        = generates interest rate from r_params, K, and L
        get_w        = generates real wage from w_params, K, and L
        get_cvec_ss  = generates consumption vector and c_constr from
                       r, w, and bvec
        get_Y        = generates aggregate output (GDP)
        get_C        = generates aggregate consumption
        get_b_errors = generates vector of dynamic Euler errors that
                       characterize lifetime savings decisions

    Objects in function:
        start_time   = scalar, current processor time in seconds (float)
        eul_objs     = length 9 tuple,
                       (S, beta, sigma, chi_b, A, alpha, delta, L, nvec)
        b_ss         = [S-1,] vector, steady state distribution of
                       savings
        K_ss         = scalar > 0, steady-state aggregate capital stock
        K_constr     = boolean, =True if K<=0 for given bvec
        r_params     = length 3 tuple, parameters for r-function
                       [A, alpha, delta]
        r_ss         = scalar > 0, steady-state interest rate (real
                       return on savings)
        w_params     = length 2 tuple, parameters for w-function
                       (A, alpha)
        w_ss         = scalar > 0, steady-state real wage
        c_ss         = [S,] vector, steady-state consumption c_s for
                       each age-s agent
        c_constr     = [S,] boolean vector, =True if c<=0 for given
                       bvec
        Y_params     = length 2 tuple, production function parameters
                       (A, alpha)
        Y_ss         = scalar > 0, steady-state aggregate output (GDP)
        C_ss         = scalar > 0, steady-state aggregate consumption
        b_err_params = length 3 tuple, parameters for Euler errors
                       (S, beta, sigma)
        EulErr_ss    = [S-1,] vector, vector of steady-state Euler
                       errors
        elapsed_time = scalar, time to compute SS solution (seconds)
        svec         = [S+1,] vector, age-s indices from 1 to S+1
        b_ss0        = [S+1,] vector, age-s wealth levels including
                       b_1=0

    Returns: b_ss, c_ss, w_ss, r_ss, BQ_ss, K_ss, EulErr_ss,
             elapsed_time
    '''
    start_time = time.clock()
    S, beta, sigma, chi_b, A, alpha, delta, L, SS_tol = params
    eul_objs = (S, beta, sigma, chi_b, A, alpha, delta, L, nvec)
    b_ss = opt.fsolve(EulerSys, b_guess, args=(eul_objs),
                      xtol=SS_tol)
    # Generate other steady-state values and Euler equations
    K_ss, K_constr = get_K(b_ss)
    r_params = (A, alpha, delta)
    r_ss = get_r(r_params, K_ss, L)
    w_params = (A, alpha)
    w_ss = get_w(w_params, K_ss, L)
    BQ_ss = get_BQ(r_ss, b_ss[-1])
    c_ss, c_constr = get_cvec_ss(S, r_ss, w_ss, BQ_ss, b_ss, nvec)
    Y_params = (A, alpha)
    Y_ss = get_Y(Y_params, K_ss, L)
    C_ss = get_C(c_ss)
    b_err_params = (beta, sigma, chi_b, b_ss[-1])
    EulErr_ss = get_b_errors(b_err_params, r_ss, c_ss, c_constr,
                bsp1_constr=np.zeros(S, dtype=bool), diff=True)
    elapsed_time = time.clock() - start_time
    if graphs == True:
        # Plot steady-state distribution of savings
        svec = np.linspace(1, S+1, S+1)
        b_ss0 = np.append([0], b_ss)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(svec, b_ss0)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Steady-state distribution of savings')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual savings $\bar{b}_{s}$')
        # plt.savefig('b_ss_Sec2')
        plt.show()

        # Plot steady-state distribution of consumption
        fig, ax = plt.subplots()
        plt.plot(svec[:-1], c_ss)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Steady-state distribution of consumption')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual consumption $\bar{c}_{s}$')
        # plt.savefig('c_ss_Sec2')
        plt.show()

    return (b_ss, c_ss, w_ss, r_ss, BQ_ss, K_ss, Y_ss, C_ss, EulErr_ss,
        elapsed_time)
