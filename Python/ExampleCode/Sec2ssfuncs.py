'''
------------------------------------------------------------------------
Last updated 7/16/2015

All the functions for the SS computation from Section 2.
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

def feasible(params, bvec):
    '''
    Determines whether a particular guess for the steady-state
    distribution of savings is feasible, first in terms of K>0, then in
    terms of c_s>0 for all s

    Inputs:
        params = [4,] vector, [S, A, alpha, delta]
        S      = integer in [3,80], number of periods in a life
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital
        bvec   = [S-1,] vector, initial guess for distribution of
                 savings b_{s+1}

    Functions called:
        get_K       = generates aggregate capital stock from bvec
        get_L       = generates aggregate labor from nvec
        get_r       = generates interest rate from r_params, K, and L
        get_w       = generates real wage from w_params, K, and L
        get_cvec_ss = generates consumption vector and c_constr from r,
                      w, and bvec

    Objects in function:
        GoodGuess = boolean, =True if initial steady-state guess is
                    feasible
        K         = scalar > 0, aggregate capital stock
        K_constr  = boolean, =True if K<=0 for given bvec
        c_constr  = [S,] boolean vector, =True if c<=0 for given bvec
        L         = scalar>0, aggregate labor
        r_params  = [3,] vector, parameters for r-function
                    [A, alpha, delta]
        r         = scalar > 0, interest rate (real return on savings)
        w_params  = [2,] vector, parameters for w-function [A, alpha]
        w         = scalar > 0, real wage
        cvec      = [S,] vector, consumption c_s for each age-s agent

    Returns: GoodGuess, K_constr, c_constr
    '''
    S, A, alpha, delta = params
    GoodGuess = True
    # Check K
    K, K_constr = get_K(bvec)
    if K_constr == True:
        GoodGuess = False
    # Check cvec if K has no problems
    c_constr = np.zeros(S, dtype=bool)
    if K_constr == False:
        L = get_L(np.ones(S))
        r_params = np.array([A, alpha, delta])
        r = get_r(r_params, K, L)
        w_params = np.array([A, alpha])
        w = get_w(w_params, K, L)
        cvec, c_constr = get_cvec_ss(S, r, w, bvec)
    if c_constr.max() == True:
        GoodGuess = False
    return GoodGuess, K_constr, c_constr


def get_L(nvec):
    '''
    Generates aggregate labor L from distribution of individual labor
    supply

    Inputs:
        nvec = [S,] vector, distribution of labor supply n_s

    Functions called: None

    Objects in function:
        L = scalar, aggregate labor

    Returns: L
    '''
    L = nvec.sum()
    return L


def get_K(bvec):
    '''
    Generates aggregate capital stock K from distribution of individual
    savings

    Inputs:
        bvec = [S-1,] vector, distribution of savings b_{s+1}

    Functions called: None

    Objects in function:
        K_constr = boolean, =True if K<=0 for given bvec
        K        = scalar, aggregate capital stock

    Returns: K, K_constr
    '''
    K_constr = False
    K = bvec.sum()
    if K <= 0:
        print 'b matrix and/or parameters resulted in K<=0'
        K_constr = True
    return K, K_constr


def get_Y(params, K, L):
    '''
    Generates aggregate output Y

    Inputs:
        params = [2,] vector, production function parameters [A, alpha]
        A      = scalar > 0, total factor productivity of production
                 function
        alpha  = scalar in (0,1), capital share of income in production
                 function
        K      = scalar > 0, aggregate capital stock
        L      = scalar > 0, aggregate labor

    Functions called: None

    Objects in function:
        Y = scalar > 0, aggregate output (GDP)

    Returns: Y
    '''
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y


def get_C(cvec):
    '''
    Generates aggregate consumption C

    Inputs:
        cvec = [S,] vector, distribution of consumption c_s

    Functions called: None

    Objects in function:
        C = scalar > 0, aggregate consumption

    Returns: C
    '''
    C = cvec.sum()
    return C


def get_r(params, K, L):
    '''
    Generates real interest rate r from parameters, aggregate capital
    stock K, and aggregate labor L

    Inputs:
        params = [3,] vector, [A, alpha, delta]
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
        params = [2,] vector, [A, alpha]
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
        w = scalar > 0 or [T+S-1,] vector, real wage or time path of
            real wage

    Returns: w
    '''
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)
    return w


def get_cvec_ss(S, r, w, bvec):
    '''
    Generates vector of consumptions from distribution of individual
    savings and the interest rate and the real wage

    Inputs:
        S    = integer in [3,80], number of periods an individual lives
        r    = scalar > 0, interest rate
        w    = scalar > 0, real wage
        bvec = [S-1,] vector, distribution of savings b_{s+1}.

    Functions called: None

    Objects in function:
        c_constr = [S,] boolean vector, =True if element c_s <= 0
        b_s      = [S,] vector, 0 in first element and bvec in last
                   S-1 elements
        b_sp1    = [S,] vector, bvec in first S-1 elements and 0 in
                   last element
        cvec     = [S,] vector, consumption by age c_s

    Returns: cvec, c_constr
    '''
    c_constr = np.zeros(S, dtype=bool)
    b_s = np.append([0], bvec)
    b_sp1 = np.append(bvec, [0])
    cvec = (1 + r) * b_s + w - b_sp1
    if cvec.min() <= 0:
        print 'initial guesses and/or parameters created c<=0 for some agent(s)'
        c_constr = cvec <= 0
    return cvec, c_constr


def get_b_errors(params, r, cvec, c_constr, diff):
    '''
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings

    Inputs:
        params   = [2,] vector, [beta, sigma]
        S      = integer in [3,80], number of periods an individual
                 lives
        beta     = scalar in [0,1), discount factor
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        cvec     = [p,] vector, distribution of consumption by age c_p
        c_constr = [p,] boolean vector, =True if c<=0 for given bvec
        diff     = boolean, =True if use simple difference Euler
                   errors. Use percent difference errors otherwise.

    Functions called: None

    Objects in function:
        mu_c     = [p-1,] vector, marginal utility of current
                   consumption
        mu_cp1   = [p-1,] vector, marginal utility of next period
                   consumption
        b_errors = [p-1,] vector, Euler errors with errors = 0
                   characterizing optimal savings bvec

    Returns: b_errors
    '''
    beta, sigma = params
    cvec[c_constr] = 9999. # Each consumption must be positive to
                           # generate marginal utilities
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    if diff == True:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
        b_errors[c_constr[:-1]] = 9999.
        b_errors[c_constr[1:]] = 9999.
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        b_errors[c_constr[:-1]] = 9999. / 100
        b_errors[c_constr[1:]] = 9999. / 100
    return b_errors


def EulerSys(bvec, ints, params):
    '''
    Generates vector of all Euler errors that characterize all
    optimal lifetime decisions

    Inputs:
        bvec   = [S-1,] vector, distribution of savings b_{s+1}
        ints   = [1,] vector, integer parameters [S]
        params = [5,] vector, [beta, sigma, A, alpha, delta]
        S      = integer in [3,80], number of periods an individual
                 lives
        beta   = scalar in [0,1), discount factor
        sigma  = scalar > 0, coefficient of relative risk aversion
        A      = scalar > 0, total factor productivity parameter in firms'
                 production function
        alpha  = scalar in (0,1), capital share of income
        delta  = scalar in [0,1], model-period depreciation rate of
                 capital

    Functions called:
        get_L        = generates aggregate labor from nvec
        get_K        = generates aggregate capital stock from bvec
        get_r        = generates interest rate from r_params, K, and L
        get_w        = generates real wage from w_params, K, and L
        get_cvec_ss  = generates consumption vector and c_constr from
                       r, w, and bvec
        get_b_errors = generates vector of dynamic Euler errors that
                       characterize lifetime savings decisions

    Objects in function:
        L            = scalar > 0, aggregate labor
        K            = scalar > 0, aggregate capital stock
        K_constr     = boolean, =True if K<=0 for given bvec
        b_err_vec    = [S-1,] vector, vector of Euler errors
        r_params     = [3,] vector, parameters for r-function
                       [A, alpha, delta]
        r            = scalar > 0, interest rate (real return on
                       savings)
        w_params     = [2,] vector, parameters for w-function
                       [A, alpha]
        w            = scalar > 0, real wage
        cvec         = [S,] vector, consumption c_s for each age-s
                       agent
        c_constr     = [S,] boolean vector, =True if c<=0 for given
                       bvec
        b_err_params = [2,] vector, parameters for Euler errors
                       [beta, sigma]

    Returns: b_errors
    '''
    S = ints
    beta, sigma, A, alpha, delta = params
    L = get_L(np.ones(S))
    K, K_constr = get_K(bvec)
    if K_constr == True:
        b_err_vec = 1000 * np.ones(S-1)
    else:
        r_params = np.array([A, alpha, delta])
        r = get_r(r_params, K, L)
        w_params = np.array([A, alpha])
        w = get_w(w_params, K, L)
        cvec, c_constr = get_cvec_ss(S, r, w, bvec)
        b_err_params = np.array([beta, sigma])
        b_err_vec = get_b_errors(b_err_params, r, cvec, c_constr,
                                 diff=True)
    return b_err_vec


def SS(ints, params, b_guess, graphs):
    '''
    Generates all endogenous steady-state objects

    Inputs:
        ints    = [1,] vector, integer parameters [S]
        S       = integer in [3,80], number of periods an individual
                  lives
        params  = [6,] vector, [beta, sigma, A, alpha, delta, SS_tol]
        beta    = scalar in [0,1), discount factor
        sigma   = scalar > 0, coefficient of relative risk aversion
        A       = scalar > 0, total factor productivity parameter in
                  firms' production function
        alpha   = scalar in (0,1), capital share of income
        delta   = scalar in [0,1], model-period depreciation rate of
                  capital
        SS_tol  = scalar > 0, tolerance level for steady-state fsolve
        b_guess = [S-1,] vector, initial guess for the distribution
                  of savings b_{s+1}
        graphs  = boolean, =True if want graphs of steady-state objects

    Functions called:
        get_L        = generates aggregate labor from nvec
        get_K        = generates aggregate capital stock from bvec
        get_r        = generates interest rate from r_params, K, and L
        get_w        = generates real wage from w_params, K, and L
        get_cvec_ss  = generates consumption vector and c_constr from
                       r, w, and bvec
        get_b_errors = generates vector of dynamic Euler errors that
                       characterize lifetime savings decisions

    Objects in function:
        eul_ints     = [1,] vector, integer parameters for EulerSys [S]
        eul_params   = [5,] vector, [beta, sigma, A, alpha, delta]
        b_ss         = [S-1,] vector, steady state distribution of
                       savings
        L_ss         = scalar > 0, steady-state aggregate labor
        K_ss         = scalar > 0, steady-state aggregate capital stock
        K_constr     = boolean, =True if K<=0 for given bvec
        r_params     = [3,] vector, parameters for r-function
                       [A, alpha, delta]
        r_ss         = scalar > 0, steady-state interest rate (real
                       return on savings)
        w_params     = [2,] vector, parameters for w-function
                       [A, alpha]
        w_ss         = scalar > 0, steady-state real wage
        c_ss         = [S,] vector, steady-state consumption c_s for
                       each age-s agent
        c_constr     = [S,] boolean vector, =True if c<=0 for given
                       bvec
        b_err_params = [3,] vector, parameters for Euler errors
                       [S, beta, sigma]
        EulErr_ss    = [S-1,] vector, vector of steady-state Euler
                       errors
        svec         = [S,] vector, age-s indices from 1 to S
        b_ss0        = [S,] vector, age-s wealth levels including b_1=0

    Returns: b_ss, c_ss, w_ss, r_ss, K_ss, EulErr_ss
    '''
    S = ints
    beta, sigma, A, alpha, delta, SS_tol = params
    eul_ints = np.array([S])
    eul_params = np.array([beta, sigma, A, alpha, delta])
    b_ss = opt.fsolve(EulerSys, b_guess, args=(eul_ints, eul_params),
                      xtol=SS_tol)
    # Generate other steady-state values and Euler equations
    L_ss = get_L(np.ones(S))
    K_ss, K_constr = get_K(b_ss)
    r_params = np.array([A, alpha, delta])
    r_ss = get_r(r_params, K_ss, L_ss)
    w_params = np.array([A, alpha])
    w_ss = get_w(w_params, K_ss, L_ss)
    c_ss, c_constr = get_cvec_ss(S, r_ss, w_ss, b_ss)
    b_err_params = np.array([beta, sigma])
    EulErr_ss = get_b_errors(b_err_params, r_ss, c_ss, c_constr, diff=True)
    if graphs == True:
        # Plot steady-state distribution of savings
        svec = np.linspace(1, S, S)
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
        plt.savefig('b_ss_Sec2')
        plt.show()

        # Plot steady-state distribution of consumption
        fig, ax = plt.subplots()
        plt.plot(svec, c_ss)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Steady-state distribution of consumption')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual consumption $\bar{c}_{s}$')
        plt.savefig('c_ss_Sec2')
        plt.show()

    return b_ss, c_ss, w_ss, r_ss, K_ss, EulErr_ss
