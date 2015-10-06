'''
------------------------------------------------------------------------
Last updated 9/29/2015

All the functions for the TPI computation.
    get_cvec_lf
    LfEulerSys
    paths_life
    get_cbepath
    TPI
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import ssfuncs as ssf
reload(ssf)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import sys

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def get_cvec_lf(params, rpath, wpath, BQpath, nvec, bvec):
    '''
    Generates vector of remaining lifetime consumptions from individual
    savings, and the time path of interest rates and the real wages

    Inputs:
        params = length 2 tuple, parameters (p, S)
        p      = integer in [2,80], number of periods remaining in
                 individual life
        S      = integer in [3,80], number of periods in individual life
        rpath  = [p,] vector, remaining interest rates
        wpath  = [p,] vector, remaining wages
        BQpath = [p,] vector, remaining total bequests
        nvec   = [p,] vector, remaining exogenous labor supply
        bvec   = [p+1,] vector, remaining savings including initial
                 savings

    Functions called: None

    Objects in function:
        c_constr = [p,] boolean vector, =True if element c_s <= 0
        b_s      = [p,] vector, bvec
        b_sp1    = [p,] vector, last p-1 elements of bvec and 0 in last
                   element
        cvec     = [p,] vector, remaining consumption by age c_s

    Returns: cvec, c_constr
    '''
    p, S = params
    c_constr = np.zeros(p, dtype=bool)
    b_s = bvec[:-1]
    b_sp1 = bvec[1:]
    cvec = (1 + rpath) * b_s + wpath * nvec + BQpath / S - b_sp1
    if cvec.min() <= 0:
        # print 'initial guesses and/or parameters created c<=0 for some agent(s)'
        c_constr = cvec <= 0
    return cvec, c_constr



def LfEulerSys(bvec, *objs):
    '''
    Generates vector of all Euler errors for a given bvec, which errors
    characterize all optimal lifetime decisions

    Inputs:
        bvec       = [p,] vector, remaining lifetime savings decisions
                     where p is the number of remaining periods
        objs       = length 10 tuple, (p, S, beta, sigma, chi_b,
                     beg_wealth, nvec, rpath, wpath, BQpath)
        p          = integer in [2,S], remaining periods in life
        beta       = scalar in [0,1), discount factor
        sigma      = scalar > 0, coefficient of relative risk aversion
        beg_wealth = scalar, wealth at the beginning of first age
        nvec       = [p,] vector, remaining exogenous labor supply
        rpath      = [p,] vector, interest rates over remaining life
        wpath      = [p,] vector, wages rates over remaining life

    Functions called:
        get_cvec_lf
        c4ssf.get_b_errors

    Objects in function:
        bvec2        = [p, ] vector, remaining savings including initial
                       savings
        clf_params   = length 2 tuple, parameters for get_cvec_lf (p, S)
        cvec         = [p, ] vector, remaining lifetime consumption
                       levels implied by bvec2
        c_constr     = [p, ] boolean vector, =True if c_{s,t}<=0
        b_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_err_vec    = [p-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_err_vec
    '''
    p, S, beta, sigma, chi_b, beg_wealth, nvec, rpath, wpath, BQpath = \
        objs
    bvec2 = np.append(beg_wealth, bvec)
    clf_params = (p, S)
    cvec, c_constr = get_cvec_lf(clf_params, rpath, wpath, BQpath, nvec,
                     bvec2)
    b_err_params = (beta, sigma, chi_b, bvec[-1])
    bsp1_constr = np.zeros(p, dtype=bool)
    if bvec[-1] <= 0:
        bsp1_constr[-1] = True
    if p == 1 and bvec[-1] > 0:
        b_err_vec = chi_b * (bvec[-1] ** (-sigma)) - (cvec[-1] **
            (-sigma))
    elif p == 1 and bvec[-1] <= 0:
        b_err_vec = 9999.
    elif p > 1:
        b_err_vec = ssf.get_b_errors(b_err_params, rpath[1:], cvec,
                                       c_constr, bsp1_constr, diff=True)
    return b_err_vec


def paths_life(params, beg_age, beg_wealth, nvec, rpath, wpath, BQpath,
               b_init):
    '''
    Solve for the remaining lifetime savings decisions of an individual
    who enters the model at age beg_age, with corresponding initial
    wealth beg_wealth.

    Inputs:
        params     = length 5 tuple, (S, beta, sigma, chi_b, TPI_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        chi_b      = scalar > 0, scale parameter on utility of bequests
        TPI_tol    = scalar > 0, tolerance level for fsolve's in TPI
        beg_age    = integer in [1,S-1], beginning age of remaining life
        beg_wealth = scalar, beginning wealth at beginning age
        nvec       = [S-beg_age+1,] vector, remaining exogenous labor
                     supplies
        rpath      = [S-beg_age+1,] vector, remaining lifetime interest
                     rates
        wpath      = [S-beg_age+1,] vector, remaining lifetime wages
        BQpath     = [S-beg_age+1,] vector, remaining lifetime total
                     bequests
        b_init     = [S-beg_age,] vector, initial guess for remaining
                     lifetime savings

    Functions called:
        LfEulerSys
        get_cvec_lf
        c4ssf.get_b_errors

    Objects in function:
        p            = integer in [1,S], remaining periods in life
        b_guess      = [p,] vector, initial guess for lifetime savings
                       decisions
        eullf_objs   = length 10 tuple, objects to be passed in to
                       LfEulerSys (p, S, beta, sigma, chi_b, beg_wealth,
                       nvec, rpath, wpath, BQpath)
        bpath        = [p,] vector, optimal remaining lifetime savings
                       decisions
        clf_params   = length 2 tuple, parameters for get_cvec_lf
        cpath        = [p,] vector, optimal remaining lifetime
                       consumption decisions
        c_constr     = [p,] boolean vector, =True if c_{p}<=0,
        b_err_params = length 2 tuple, parameters to pass into
                       c4ssf.get_b_errors (beta, sigma)
        rpath2       = [p+1,] vector, rpath with 0 on the end so we can
                       pass it in to get_b_errors for p=1 which does not
                       require an interest rate
        b_err_vec    = [p,] vector, Euler errors associated with
                       optimal savings decisions

    Returns: bpath, cpath, b_err_vec
    '''
    S, beta, sigma, chi_b, TPI_tol = params
    p = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(rpath) != p:
        #print len(rpath), S-beg_age+1
        sys.exit("Beginning age and length of rpath do not match.")
    if len(wpath) != p:
        sys.exit("Beginning age and length of wpath do not match.")
    if len(nvec) != p:
        sys.exit("Beginning age and length of nvec do not match.")
    b_guess = 1.01 * b_init
    eullf_objs = (p, S, beta, sigma, chi_b, beg_wealth, nvec, rpath,
                 wpath, BQpath)
    bpath = opt.fsolve(LfEulerSys, b_guess, args=(eullf_objs),
                       xtol=TPI_tol)
    clf_params = (p, S)
    cpath, c_constr = get_cvec_lf(clf_params, rpath, wpath, BQpath, nvec,
                                  np.append(beg_wealth, bpath))
    b_err_params = (beta, sigma, chi_b, bpath[-1])
    bsp1_constr = np.zeros(p, dtype=bool)
    if bpath[-1] <= 0:
        bsp1_constr[-1] = True
    if beg_age == S and bpath[-1] > 0:
        b_err_vec = chi_b * (bpath[-1] ** (-sigma)) - (cpath[-1] **
                    (-sigma))
    elif beg_age == S and bpath[-1] <=0:
        b_err_vec = 9999.
    elif beg_age < S:
        b_err_vec = ssf.get_b_errors(b_err_params, rpath[1:], cpath,
                                       c_constr, bsp1_constr, diff=True)
    return bpath, cpath, b_err_vec


def get_cbepath(params, nvec, rpath, wpath, BQpath, Gamma1, b_ss):
    '''
    Generates matrices for the time path of the distribution of
    individual savings, individual consumption, and the Euler errors
    associated with the savings decisions.

    Inputs:
        params  = length 6 tuple, (S, T, beta, sigma, chi_b, TPI_tol)
        S       = integer in [3,80], number of periods an individual
                  lives
        T       = integer > S, number of time periods until steady state
        beta    = scalar in [0,1), discount factor for each model period
        sigma   = scalar > 0, coefficient of relative risk aversion
        chi_b   = scalar > 0, scale parameter on utility of bequests
        TPI_tol = scalar > 0, tolerance level for fsolve's in TPI
        nvec    = [S,] vector, exogenous labor supply n_{s,t}
        rpath   = [T+S-1,] vector, equilibrium time path of the interest
                  rate
        wpath   = [T+S-1,] vector, equilibrium time path of the real
                  wage
        BQpath  = [T+S-1,] vector, equilibrium time path of total
                  bequests
        Gamma1  = [S,] vector, initial period savings distribution
        b_ss    = [S,] vector, steady-state savings distribution

    Functions called:
        paths_life

    Objects in function:
        cpath      = [S, T+S-2] matrix,
        bpath      = [S, T+S-1] matrix,
        EulErrPath = [S, T+S-1] matrix,
        pl_params  = length 5 tuple, parameters to pass into paths_life
                     (S, beta, sigma, chi_b, TPI_tol)
        p          = integer >= 1, represents number of periods
                     remaining in a lifetime, used to solve incomplete
                     lifetimes
        b_guess    = [p,] vector, initial guess for remaining lifetime
                     savings, taken from previous cohort's choices
        bveclf     = [p,] vector, optimal remaining lifetime savings
                     decisions
        cveclf     = [p,] vector, optimal remaining lifetime consumption
                     decisions
        b_err_veclf = [p,] vector, Euler errors associated with
                      optimal remaining lifetime savings decisions
        DiagMask    = [p, p] boolean identity matrix

    Returns: cpath, bpath, EulErrPath
    '''
    S, T, beta, sigma, chi_b, TPI_tol = params
    cpath = np.zeros((S, T+S-2))
    bpath = np.append(Gamma1.reshape((S, 1)), np.zeros((S, T+S-2)),
            axis=1)
    EulErrPath = np.zeros((S, T+S-1))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    pl_params = (S, beta, sigma, chi_b, TPI_tol)
    for p in xrange(1, S):
        # b_guess = b_ss[-p+1:]
        b_guess = np.diagonal(bpath[S-p:, :p])
        bveclf, cveclf, b_err_veclf = paths_life(pl_params, S-p+1,
            Gamma1[S-p-1], nvec[-p:], rpath[:p], wpath[:p], BQpath[:p],
            b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        DiagMask = np.eye(p, dtype=bool)
        bpath[S-p:, 1:p+1] = DiagMask * bveclf + bpath[S-p:, 1:p+1]
        cpath[S-p:, :p] = DiagMask * cveclf + cpath[S-p:, :p]
        EulErrPath[S-p:, 1:p+1] = (DiagMask * b_err_veclf +
                                EulErrPath[S-p:, 1:p+1])
    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    DiagMask = np.eye(S, dtype=bool)
    for t in xrange(1, T): # Go from periods 1 to T-1
        # b_guess = b_ss
        b_guess = np.diagonal(bpath[:, t-1:t+S-1])
        bveclf, cveclf, b_err_veclf = paths_life(pl_params, 1, 0,
            nvec, rpath[t-1:t+S-1], wpath[t-1:t+S-1], BQpath[t-1:t+S-1],
            b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        bpath[:, t:t+S] = DiagMask * bveclf + bpath[:, t:t+S]
        cpath[:, t-1:t+S-1] = DiagMask * cveclf + cpath[:, t-1:t+S-1]
        EulErrPath[:, t:t+S] = (DiagMask * b_err_veclf +
                                 EulErrPath[:, t:t+S])

    return cpath, bpath, EulErrPath


def TPI(params, Kpath_init, BQpath_init, Gamma1, nvec, Lpath, b_ss,
  graphs):
    '''
    Generates steady-state time path for all endogenous objects from
    initial state (K1, BQ1, Gamma1) to the steady state.

    Inputs:
        params      = length 17 tuple, (S, T, beta, sigma, chi_b, L, A,
                      alpha, delta, K1, K_ss, BQ_ss, C_ss maxiter_TPI,
                      mindist_TPI, xi, TPI_tol)
        S           = integer in [3,80], number of periods an individual
                      lives
        T           = integer > S, number of time periods until steady
                      state
        beta        = scalar in [0,1), discount factor for each model
                      period
        sigma       = scalar > 0, coefficient of relative risk aversion
        chi_b       = scalar > 0, scale parameter on utility of bequests
        L           = scalar > 0, exogenous aggregate labor
        A           = scalar > 0, total factor productivity parameter in
                      firms' production function
        alpha       = scalar in (0,1), capital share of income
        delta       = scalar in [0,1], model-period depreciation rate of
                      capital
        K1          = scalar > 0, initial period aggregate capital stock
        K_ss        = scalar > 0, steady-state aggregate capital stock
        BQ_ss       = scalar > 0, steady-state total bequests
        maxiter_TPI = integer >= 1, Maximum number of iterations for TPI
        mindist_TPI = scalar > 0, Convergence criterion for TPI
        xi          = scalar in (0,1], TPI path updating parameter
        TPI_tol     = scalar > 0, tolerance level for fsolve's in TPI
        Kpath_init  = [T+S-1,] vector, initial guess for the time path
                      of the aggregate capital stock
        BQpath_init = [T+S-1,] vector, initial guess for the time path
                      of total bequests
        Gamma1      = [S,] vector, initial period savings distribution
        nvec        = [S,] vector, exogenous labor supply n_{s,t}
        Lpath       = [T+S-1,] vector, exogenous time path for aggregate
                      labor
        b_ss        = [S,] vector, steady-state savings distribution
        graphs      = boolean, =True if want graphs of TPI objects

    Functions called:
        c8ssf.get_r
        c8ssf.get_w
        get_cbepath
        c4ssf.get_K

    Objects in function:
        start_time   = scalar, current processor time in seconds (float)
        iter_TPI     = integer >= 0, current iteration of TPI
        dist_TPI     = scalar >= 0, distance measure for fixed point
        Kpath_new    = [T+S-2,] vector, new path of the aggregate
                       capital stock implied by household and firm
                       optimization
        BQpath_new   = [T+S-1,] vector, new path of total bequests
                       implied by household and firm optimization
        r_params     = length 3 tuple, parameters passed in to get_r
        w_params     = length 2 tuple, parameters passed in to get_w
        cbe_params   = length 6 tuple. parameters passed in to
                       get_cbepath
        rpath        = [T+S-2,] vector, equilibrium time path of the
                       interest rate
        wpath        = [T+S-2,] vector, equilibrium time path of the
                       real wage
        cpath        = [S, T+S-2] matrix, equilibrium time path values
                       of individual consumption c_{s,t}
        bpath        = [S, T+S-1] matrix, equilibrium time path values
                       of individual savings b_{s+1,t+1}
        EulErrPath   = [S, T+S-1] matrix, equilibrium time path values
                       of Euler errors corresponding to individual
                       savings b_{s+1,t+1} (first column is zeros)
        Kpath_constr = [T+S-2,] boolean vector, =True if K_t<=0
        Kpath        = [T+S-2,] vector, equilibrium time path of the
                       aggregate capital stock
        Y_params     = length 2 tuple, parameters to be passed to get_Y
        Ypath        = [T+S-2,] vector, equilibrium time path of
                       aggregate output (GDP)
        Cpath        = [T+S-2,] vector, equilibrium time path of
                       aggregate consumption
        elapsed_time = scalar, time to compute TPI solution (seconds)

    Returns: bpath, cpath, BQpath, wpath, rpath, Kpath, Ypath, Cpath,
             EulErrpath, elapsed_time
    '''
    start_time = time.clock()
    (S, T, beta, sigma, chi_b, L, A, alpha, delta, K1, K_ss, BQ_ss,
        C_ss, maxiter_TPI, mindist_TPI, xi, TPI_tol) = params
    iter_TPI = int(0)
    dist_TPI = 10.
    Kpath_new = Kpath_init
    BQpath_new = BQpath_init
    r_params = (A, alpha, delta)
    w_params = (A, alpha)
    cbe_params = (S, T, beta, sigma, chi_b, TPI_tol)

    while (iter_TPI < maxiter_TPI) and (dist_TPI >= mindist_TPI):
        iter_TPI += 1
        Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
        BQpath_init = xi * BQpath_new + (1 - xi) * BQpath_init
        rpath = ssf.get_r(r_params, Kpath_init, Lpath)
        wpath = ssf.get_w(w_params, Kpath_init, Lpath)
        cpath, bpath, EulErrPath = get_cbepath(cbe_params, nvec, rpath,
                                   wpath, BQpath_init, Gamma1, b_ss)
        Kpath_new = np.zeros(T+S-2)
        Kpath_new[:T], Kpath_constr = ssf.get_K(bpath[:, :T])
        Kpath_new[T:] = K_ss
        Kpath_constr = np.append(Kpath_constr, np.zeros(S-1, dtype=bool))
        Kpath_new[Kpath_constr] = 1
        BQpath_new = np.zeros(T+S-2)
        BQpath_new[:T] = ssf.get_BQ(rpath[:T], bpath[S-1, :T])
        BQpath_new[T:] = BQ_ss

        # Check the distance of Kpath_new and BQpath_new
        Kdist_TPI = (Kpath_new[1:T] - Kpath_init[1:T]) / Kpath_init[1:T]
        BQdist_TPI = (BQpath_new[1:T] - BQpath_init[1:T]) / BQpath_init[1:T]
        dist_TPI = np.absolute(np.append(Kdist_TPI, BQdist_TPI)).max()
        print ('iter: ', iter_TPI, ', dist: ', dist_TPI, ', max Eul err: ',
            np.absolute(EulErrPath).max())

    if iter_TPI == maxiter_TPI and dist_TPI > mindist_TPI:
        print 'TPI reached maxiter and did not converge.'
    elif iter_TPI == maxiter_TPI and dist_TPI <= mindist_TPI:
        print 'TPI converged in the last iteration. Should probably increase maxiter_TPI.'
    Kpath = Kpath_new
    BQpath = BQpath_new
    Y_params = (A, alpha)
    Ypath = ssf.get_Y(Y_params, Kpath, Lpath)
    Cpath = np.zeros(T+S-2)
    Cpath[:T-1] = ssf.get_C(cpath[:, :T-1])
    Cpath[T-1:] = C_ss
    elapsed_time = time.clock() - start_time

    if graphs == True:
        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T+S-2, T+S-2)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Kpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate capital stock')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate capital $K_{t}$')
        # plt.savefig('Kt_Chap8')
        plt.show()

        # Plot time path of total bequests
        tvec = np.linspace(1, T+S-2, T+S-2)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, BQpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for total bequests')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Total bequests $BQ_{t}$')
        # plt.savefig('BQt_Chap8')
        plt.show()

        # Plot time path of aggregate output (GDP)
        tvec = np.linspace(1, T+S-2, T+S-2)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Ypath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate output (GDP)')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate output $Y_{t}$')
        # plt.savefig('Yt_Chap8')
        plt.show()

        # Plot time path of aggregate consumption
        tvec = np.linspace(1, T+S-2, T+S-2)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Cpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate consumption')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate consumption $C_{t}$')
        # plt.savefig('Ct_Chap8')
        plt.show()

        # Plot time path of real wage
        tvec = np.linspace(1, T+S-2, T+S-2)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, wpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for real wage')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real wage $w_{t}$')
        # plt.savefig('wt_Chap8')
        plt.show()

        # Plot time path of real interest rate
        tvec = np.linspace(1, T+S-2, T+S-2)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, rpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for real interest rate')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real interest rate $r_{t}$')
        # plt.savefig('rt_Chap8')
        plt.show()

        # Plot time path of individual savings distribution
        tgrid = np.linspace(1, T, T)
        sgrid = np.linspace(2, S+1, S)
        tmat, smat = np.meshgrid(tgrid, sgrid)
        cmap_bp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual savings $b_{s,t}$')
        strideval = max(int(1), int(round(S/10)))
        ax.plot_surface(tmat, smat, bpath[:, :T], rstride=strideval,
            cstride=strideval, cmap=cmap_bp)
        # plt.savefig('bpath_Chap8')
        plt.show()

        # Plot time path of individual consumption distribution
        tgrid = np.linspace(1, T-1, T-1)
        sgrid = np.linspace(1, S, S)
        tmat, smat = np.meshgrid(tgrid, sgrid)
        cmap_cp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual consumption $c_{s,t}$')
        strideval = max(int(1), int(round(S/10)))
        ax.plot_surface(tmat, smat, cpath[:, :T-1], rstride=strideval,
            cstride=strideval, cmap=cmap_cp)
        # plt.savefig('cpath_Chap8')
        plt.show()

    return (bpath, cpath, BQpath, wpath, rpath, Kpath, Ypath, Cpath,
        EulErrPath, elapsed_time)
