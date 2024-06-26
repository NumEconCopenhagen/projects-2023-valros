import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import sympy as sm
from IPython.display import display
from types import SimpleNamespace
import numba as nb

# functions for problem 1
def analytical_sol(do_print = False):

    # a. define symbols
    alpha = sm.symbols('alpha', positive=True)
    nu = sm.symbols('nu', positive=True)
    kappa = sm.symbols('kappa', positive=True)
    tau = sm.symbols('tau', positive=True)
    C = sm.symbols('C')
    G = sm.symbols('G', positive=True)
    L = sm.symbols('L')
    w = sm.symbols('w', positive=True)
    wtilde = sm.symbols('wtilde', positive=True)
    Lstar = sm.symbols('L^*')

    # b. define utility function and budget constraint
    util = sm.log(C**alpha * G**(1-alpha)) - nu * (L**2)/2
    cons = kappa + (1-tau)*w*L  

    # c. substitute (1-tau)*w for wtilde and insert into utility function
    cons = cons.subs(w*(1-tau), wtilde)
    util = util.subs(C, cons)    

    # d. differentiate utility function wrt. L and set equal to 0
    util_diff = sm.diff(util, L)
    sol = sm.solve(util_diff, L)[1] # choosing the second solution as the first is negative
    sol_expanded = sol.subs(wtilde, (1-tau)*w)

    # e. print solution
    if do_print:
        display(sm.Eq(Lstar, sol))
    return sol, sol_expanded

class TaxModel:
    def __init__(self):
        """ 
        Initialize class and create empty simple namespace

        arguments:
            none

        returns:
            none 
        """

        # a. create empty simple namespace
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()

    def set_values(self):
        """
        Sets baseline parameter values.

        arguments:
            none

        returns:
            none
        """

        # a. call simple namespace
        par = self.par

        # b. set baseline parameter values
        par.alpha = 0.5
        par.kappa = 1.0
        par.nu = 1 / (2 * 16**2)
        par.w = 1.0
        par.tau = 0.3

        # c. set baseline values for the extension
        par.epsilon = 1.0

    def labour_plot(self):
        """
        Plots labour supply as a function of wtilde.

        arguments:
            none

        returns:
            plot of labour supply as a function of wtilde
        """

        # a. call parmeter values and necessary symbols
        par = self.par
        alpha = sm.symbols('alpha', positive=True)
        kappa = sm.symbols('kappa', positive=True)
        wtilde = sm.symbols('wtilde', positive=True)
        nu = sm.symbols('nu', positive=True)
        
        # b. call analytical_sol function and extract solution as a function of wtilde
        sol = analytical_sol()[0]

        # c. define function for labour supply
        L_func = sm.lambdify((alpha, nu, kappa, wtilde), sol)

        # d. define array of wtilde values and calculate labour supply
        wtilde_vec = np.linspace(0.0000001, 1, 100)
        Lstar = L_func(alpha=par.alpha, kappa=par.kappa, nu=par.nu, wtilde=wtilde_vec)

        # e. plot labour supply as a function of wtilde
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(wtilde_vec, Lstar)
        ax.set_xlabel(r'$\widetilde{w}$')
        ax.set_ylabel(r'$L$')
        ax.set_title(r'Labour supply as a function of $\widetilde{w}$')
        plt.show()

    def labour_fun(self,tau):
        """
        Calculates the labour supply.

        arguments:
            tau (float): tax rate

        returns:
            labour supply
        """

        # a. call parmeter values and necessary symbols
        par = self.par

        # b. calculate labour supply and return
        return (-par.kappa + np.sqrt(par.kappa**2 + 4 * (par.alpha/par.nu) * (par.w * (1-tau))**2)) / (2 * par.w * (1-tau))
    
    def govern_fun(self,tau):
        """
        Calculates the government spending.

        arguments:
            tau (float): tax rate

        returns:
            government spending
        """

        # a. call parmeter values
        par = self.par

        # b. calculate government spending and return
        return tau * par.alpha * par.w * self.labour_fun(tau=tau)
    
    def consump_fun(self, tau):
        """
        Calculates the consumption.

        arguments:
            tau (float): tax rate

        returns:
            consumption
        """

        # a. call parmeter values
        par = self.par

        # b. calculate consumption and return
        return par.kappa + (1-tau) * par.w * self.labour_fun(tau=tau)
    
    def util_fun(self, tau):
        """
        Calculates the utility.

        arguments:
            tau (float): tax rate

        returns:
            utility
        """

        # a. call parmeter values
        par = self.par

        # b. calculate utility and return
        return np.log(self.consump_fun(tau=tau)**par.alpha * self.govern_fun(tau=tau)**(1-par.alpha)) - par.nu * (self.labour_fun(tau=tau)**2)/2
        
    def tax_plot(self):
        """
        Plots labour supply, government spending, and utility as a function of the tax rate.

        arguments:
            none

        returns:
            plot of labour supply, government spending, and utility as a function of the tax rate
        """

        # a. call parmeter values
        par = self.par

        # b. define array of tau values
        tau = np.linspace(0.001, 0.999, 500)

        # c. call labour, government spending, and utility functions
        labour = self.labour_fun(tau=tau)
        gover = self.govern_fun(tau=tau)
        util = self.util_fun(tau=tau)

        # d. plot labour supply, government spending, and utility as a function of tau
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Labor Supply, Government Consumption, and Worker Utility as Functions of the Tax Rate', fontsize = 15)

        # e. create subplots for labour supply, government spending, and utility
        # i. labour supply
        plt.subplot(3, 1, 1)
        plt.plot(tau, labour, label='Labor Supply (L)')
        plt.ylabel(r'Labor Supply (L)', size = 10)
        plt.legend(fontsize = 10, loc = 'lower center')

        # ii. government spending
        plt.subplot(3, 1, 2)
        plt.plot(tau, gover, label='Government Consumption (G)', color = 'red')
        plt.ylabel(r'Government Consumption (G)', size = 10)
        plt.legend(fontsize = 10, loc = 'lower center')

        # iii. utility
        plt.subplot(3, 1, 3)
        plt.plot(tau, util, label='Worker Utility (V)', color = 'green')
        plt.xlabel('Tax Rate ' +r'$\tau$', fontsize = 10)
        plt.ylabel(r'Worker Utility (V)', size = 10)
        plt.legend(fontsize = 10, loc = 'lower center')

        # f. adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def optimal_tax(self, do_print=False, do_plot=False):
        """
        Calculates the optimal tax rate.

        arguments:
            do_print (bool): whether to print the optimal tax rate
            do_plot (bool): whether to plot utility as a function of the tax rate

        returns:
            sol.tau (float): the optimal tax rate
        """

        # a. call solution namespace
        sol = self.sol

        # b. define objective function
        obj = lambda tau: -self.util_fun(tau=tau)

        # c. solve for the optimal tax rate using Nelder-Mead minimization
        sol.tau = optimize.minimize(obj, x0=0.5, method = 'BFGS').x[0]
        
        # d. print optimal tax rate if do_print is True
        if do_print:
            print(f'The optimal tax rate is {sol.tau:.4f}.')

        # e. plot utility as a function of the tax rate if do_plot is True
        if do_plot:
            # i. define array of tau values
            tau = np.linspace(0.001, 0.999, 500)

            # ii. call utility function
            util = self.util_fun(tau=tau)

            # iii. plot utility as a function of tau
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1,1,1)
            ax.set_title('Worker Utility as a Function of the Tax Rate', fontsize = 15)
            ax.axvline(x = sol.tau, color = "black", linestyle = '--', label = r'Optimal Tax Rate $\tau =$'+f'{sol.tau:.4f}')
            ax.plot(tau, util, label='Worker Utility (V)', color = 'green')
            plt.xlabel('Tax Rate ' +r'$\tau$', fontsize = 10)
            plt.ylabel(r'Worker Utility (V)', size = 10)
            ax.legend(fontsize = 10)

            # iv. adjust layout and show plot
            plt.tight_layout()
            plt.show()

        # f. return the optimal tax rate
        return sol.tau
    
    def labour_opt(self, sigma, rho, tau, G, do_print=False):
        """
        Solves the worker's labour problem for a given tax rate and government spending.

        arguments:
            sigma (float): elasticity of substitution
            rho (float): risk aversion 
            tau (float): tax rate
            G (float): government spending
            do_print (bool): whether to print the optimal labor supply

        returns:
            res.x[0] (float): optimal labor supply
        """

        # a. call parmeter values
        par = self.par

        # b. setup the objective function
        obj = lambda L: - ((((par.alpha*(par.kappa + (1-tau) * par.w * L)**((sigma-1)/sigma) 
                            + (1-par.alpha)*G**((sigma-1)/sigma))**(sigma/(sigma-1)))**(1-rho) - 1) 
                            / (1-rho) - par.nu * L**(1+par.epsilon) / (1+par.epsilon))

        # c. solve for the optimal labor supply using BFSG minimization
        res = optimize.minimize(obj, x0=12, bounds=[(0,24)], method = 'Nelder-Mead')

        # d. print optimal labor supply if do_print is True
        if do_print:
            print(f'The optimal labor supply is {res.x[0]:.4f}.')

        # e. return the optimal labor supply
        return res.x[0]

    def gov_opt(self, sigma, rho, tau, do_print=False):
        """
        Solves the government's problem for a given tax rate.

        arguments:
            sigma (float): elasticity of substitution
            rho (float): risk aversion 
            tau (float): tax rate
            do_print (bool): whether to print the optimal government spending

        returns:
            res.root (float): government spending
        """
        
        # a. call parmeter values
        par = self.par

        # b. setup the objective function
        obj = lambda G: G - tau * par.w * self.labour_opt(sigma=sigma, rho=rho, tau=tau, G=G)
        
        # c. solve for the optimal labor supply using brentq root scalar
        res = optimize.root_scalar(obj, bracket=(0,24), method = 'brentq')

        # d. print optimal labor supply if do_print is True
        if do_print:
            print(f'The optimal government spending is {res.root:.4f}.')

        # e. return the optimal labor supply
        return res.root

    def util_ext_fun(self, sigma, rho, tau, do_print=False):
        """
        Calculates the utility for a given tax rate.

        arguments:
            sigma (float): elasticity of substitution
            rho (float): risk aversion        
            tau (float): tax rate
            do_print (bool): whether to print the utility

        returns:
            util (float): utility
        """
        # a. call parmeter values
        par = self.par

        # b. calculate government spending and optimal labor supply
        G_opt = self.gov_opt(sigma=sigma, rho=rho, tau=tau)
        L_opt = self.labour_opt(sigma=sigma, rho=rho, tau=tau, G=G_opt)

        # c. calculate utility
        util = ((((par.alpha * (par.kappa + (1 - tau) * par.w * L_opt) ** ((sigma - 1) / sigma) 
                   + (1 - par.alpha) * G_opt ** ((sigma - 1) / sigma)) ** (sigma / (sigma - 1))) ** (1 - rho) - 1) / (1 - rho) 
                   - par.nu * L_opt ** (1 + par.epsilon) / (1 + par.epsilon))
        
        # d. print utility if do_print is True
        if do_print:
            print(f'The utility is {util:.4f}.')
        
        # e. return utility
        return util
    
    def optimal_tax_ext(self, sigma, rho, do_print=False):
        """
        Solves for the socially optimal tax rate.

        arguments:
            sigma (float): elasticity of substitution
            rho (float): risk aversion
            do_print (bool): if True, prints the optimal tax rate

        returns:
            sol.tau_ext (float): optimal tax rate
        """

        # a. call parmeter values
        sol = self.sol

        # b. setup the objective function
        obj = lambda tau: - self.util_ext_fun(sigma=sigma, rho=rho, tau=tau)

        # c. solve for the optimal tax rate using BFSG minimization
        sol.tau_ext = optimize.minimize(obj, x0=0.5, bounds=[(0,1)], method = 'Nelder-Mead').x[0]

        # d. print optimal tax rate if do_print is True
        if do_print:
            print(f'The optimal tax rate is {sol.tau_ext:.4f}.')
        
        # e. return the optimal tax rate
        return sol.tau_ext

# functions for problem 2
class HairModel:
    def __init__(self):
        """
        initializes the class

        arguments:
            None

        returns:
            None
        """

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

    def set_values(self):
        """
        sets the parameter values

        arguments:
            None

        returns:
            None
        """

        # a. call parmeter values
        par = self.par

        # c. set parameter values
        par.eta = 0.5
        par.w = 1.0
        par.rho = 0.90
        par.iota = 0.01
        par.sigma = 0.10
        par.R = (1+0.01)**(1/12)
        par.T = 120
        par.K = 10000
        
    def numerical_l(self, kappa, do_print=False):
        """
        Calculates the optimal number of hairdressers for a given kappa.

        arguments:
            kappa (float): the value of the shock
            do_print (bool): whether to print the optimal number of hairdressers

        returns:
            res.x[0] (float): optimal amount of hairdressers
        """

        # a. call parmeter values
        par = self.par

        # b. setup the objective function        
        obj = lambda l: - (kappa * l ** (1-par.eta) - par.w * l)

        # c. solve for the optimal number of hairdressers using BFSG minimization
        res = optimize.minimize(fun=obj, x0=0.5, method = 'BFGS')

        # d. print optimal number of hairdressers if do_print is True
        if do_print:
            print(f'The optimal number of hairdressers is {res.x[0]:.4f}.')

        # e. return the optimal number of hairdressers
        return res.x[0]
    
    def analytical_l(self, kappa, do_print=False):
        """
        Calculates the optimal number of hairdressers for a given kappa.

        arguments:
            kappa (float): the value of the shock
            do_print (bool): whether to print the optimal number of hairdressers
            
        returns:
            l (float): optimal amount of hairdressers
        """

        # a. call parmeter values
        par = self.par

        # b. calculate optimal number of hairdressers and return
        l = ((1-par.eta) * kappa / par.w) ** (1/par.eta)

        # c. print optimal number of hairdressers if do_print is True
        if do_print:
            print(f'The optimal number of hairdressers is {l:.4f}.')

        return l 
    
    def plot_l(self, kappa=(1.0, 2.0)):
        """
        Plots the optimal number of hairdressers for different values of kappa.

        arguments:
            include_analytical (bool): if True, includes the analytical solution

        returns:
            None
        """

        # a. setup the grid
        kappa_grid = np.linspace(kappa[0],kappa[1], 100)

        # b. setup the figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # c. plot the optimal number of hairdressers
        ax.plot(kappa_grid, [self.numerical_l(kappa) for kappa in kappa_grid], label='Numerical solution')
        ax.plot(kappa_grid, [self.analytical_l(kappa) for kappa in kappa_grid], linestyle='--', alpha=0.75, label='Analytical solution')

        # d. set labels and legend
        ax.set_xlabel(r'$\kappa$')
        ax.set_ylabel(r'$l$')
        ax.legend()
        ax.set_title('Optimal number of hairdressers for different values of $\kappa$')

        # e. show the figure
        plt.show()
    
    def kappa_path(self):
        """
        Calculates the path of kappa.

        arguments:
            None

        returns:
            sim.kappa (np.array): path of kappa
        """

        # a. set relevant parameters
        par = self.par
        sim = self.sim

        # b. initialize empty arrays for kappa 
        log_kappa = np.zeros((par.K, par.T+1))

        # c. dertermine shocks for the entire period
        np.random.seed(2023)
        eps = np.random.normal(-0.5*par.sigma**2, par.sigma, (par.K, par.T))

        # d. calculate kappa for each period
        for t in range(1,par.T+1): # as kappa is 1 in period t=0, we only need to calculate it for t=1,2,...,T+1
            log_kappa[:,t] = par.rho * log_kappa[:,t-1] + eps[:,t-1]
        sim.kappa = np.exp(log_kappa)

        # e. return the path of kappa
        return sim.kappa

# The following functions are stil part of the solution to problem 2 but are run outside the class to take advantage of numba (we could not get numba.experimental.jitclass to work)

@nb.njit(parallel=True)
def policy(eta, iota, rho, sigma, K, R, T, w, kappa, Delta=0, ext=False):
    """
    Calculates the optimal policy for a given kappa.
    Redundant arguments are included to make the function compatible with a dictionary of parameters.

    arguments:
        eta (float): productivity parameter for labour
        K (int): number of simulations
        T (int): number of periods
        w (float): wage
        kappa (np.array): path of kappa
        Delta (float): maximum difference between labour in two consecutive periods

    redundant arguments:
        iota (float): price of changing the number of hairdressers
        rho (float): persistence parameter for kappa
        sigma (float): standard deviation of the shock
        R (float): discounting factor

    returns:
        labour (np.array): optimal policy for labour
    """

    # a. initialize empty array for labour
    labour = np.zeros((K,T+1))

    # b. calculate the labour baseline for each period
    for t in range(1, T+1):
        labour[:,t] = (((1-eta)*kappa[:,t])/w)**(1/(eta)) 

    # c. adjust labour if the difference between two consecutive periods is too large
    for t in range(1, T+1):
        for k in range(0,K):
            if ext:
                mark = np.abs(labour[k,t] - labour[k,t-1]-0.0025)
            else:
                mark = np.abs(labour[k,t] - labour[k,t-1])
            if mark <= Delta:
                    labour[k,t] = labour[k,t-1]

    # d. return the optimal policy for labour
    return labour


@nb.njit(parallel=True)
def ex_ante(eta, iota, rho, sigma, K, R, T, w, kappa, labour, do_print=False):
    """
    Calculates the ex ante value of the hairsalon.
    Redundant arguments are included to make the function compatible with a dictionary of parameters.

    arguments:
        eta (float): productivity parameter for labour
        iota (float): price of changing the number of hairdressers
        K (int): number of simulations
        R (float): discounting factor
        T (int): number of periods
        w (float): wage
        kappa (np.array): path of kappa
        labour (np.array): optimal policy for labour
        do_print (bool): if True, prints the ex ante value of the hairsalon

    redundant arguments:
        rho (float): persistence parameter for kappa
        sigma (float): standard deviation of the shock

    returns:
        V (float): ex ante value of the hairsalon
    """

    # a. initialize empty arrays for iota, the single period value of salon
    iota_vec = np.zeros((K, T+1))
    single_period = np.zeros((K, T+1))

    # b. calculate cost of changing the number of hairdressers
    for k in range(0,K):
        for t in range(0,T+1):
            if labour[k,t] != labour[k,t-1]:
                iota_vec[k,t] = iota

    # c. calculate the single period value
    for t in range(1, T+1):
        single_period[:,t] = R**(-t) * (kappa[:,t]*labour[:,t]**(1-eta)-w*labour[:,t]-iota_vec[:,t])

    # d. calculate the ex ante value
    ex_post = np.sum(single_period, axis=1)
    V = np.mean(ex_post)

    # e. print the ex ante value if do_print is True
    if do_print:
        print('The ex ante value of the salon is:', V)

    # f. return the ex ante value
    return V

def optimal_delta(eta, iota, rho, sigma, K, R, T, w, kappa, interval=(0,1), ext=False, do_print=False):
    """
    Calculates the optimal delta.

    arguments:
        eta (float): productivity parameter for labour
        iota (float): price of changing the number of hairdressers
        rho (float): persistence parameter for kappa
        sigma (float): standard deviation of the shock
        K (int): number of simulations
        R (float): discounting factor
        T (int): number of periods
        w (float): wage
        kappa (np.array): path of kappa
        interval (tuple): interval for delta
        do_print (bool): if True, prints the optimal delta

    returns:
        res.x (float): optimal delta
    """


    # a. ccreate dictionary of parameters to pass to the objective function
    param = {'eta':eta, 'iota':iota, 'rho':rho, 'sigma':sigma, 'K':K, 'R':R, 'T':T, 'w':w}

    # b. create objective function for minimization
    obj = lambda delta: - ex_ante(**param, labour = policy(**param, kappa = kappa, Delta = delta, ext=ext), kappa = kappa)

    # c. minimize the objective function
    res = optimize.minimize_scalar(obj, bounds=interval, method='bounded')

    # d. print the optimal delta if do_print is True
    if do_print:
        print(f'The optimal delta is: {res.x}\n'
              f'The ex ante value of the salon is: {-res.fun}')

    # e. return the optimal delta
    return res

@nb.jit(parallel=True, forceobj=True)
def value_plot(eta, iota, rho, sigma, K, R, T, w, kappa, optimal_delta, ext=False, interval=(0,1)):
    """
    Plots the expected value of the hairsalon for different delta values.

    arguments:
        eta (float): productivity parameter for labour
        iota (float): price of changing the number of hairdressers
        rho (float): persistence parameter for kappa
        sigma (float): standard deviation of the shock
        K (int): number of simulations
        R (float): discounting factor
        T (int): number of periods
        w (float): wage
        kappa (np.array): path of kappa
        optimal_delta (float): optimal delta, can be calculated by optimal_delta()
        interval (tuple): interval for delta

    returns:
        fig (matplotlib.figure): figure with plot of the expected value of the hairsalon for different delta values
    """

    # a. create array for delta values and ex ante values
    delta = np.linspace(interval[0],interval[1], 100)
    ex_ante_vec = np.zeros(100)

    # b. calculate ex ante value for different delta values
    for i in range(0,100):
        labour = policy(eta, iota, rho, sigma, K, R, T, w, kappa=kappa, Delta=delta[i], ext=ext)
        ex_ante_vec[i] = ex_ante(eta, iota, rho, sigma, K, R, T, w, labour = labour, kappa = kappa)
        
    # c. setup the figure and plot the ex ante value
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(delta, ex_ante_vec, label='Ex ante value')
    ax.plot(optimal_delta.x, -optimal_delta.fun, 'rx', label='Optimal delta')

    # d. set labels and title and show the figure
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'$V$')
    ax.set_title('Ex ante value of the hairsalon for different values of $\Delta$')
    ax.legend()
    plt.show()


# functions for problem 3
@nb.njit()
def griewank(x):
    """
    Returns the value of the Griewank function for a given array x.

    arguments:
        x (array): the array for which to calculate the value of the function

    returns:
        float: the value of the function
    """

    return griewank_(x[0],x[1])

@nb.njit()
def griewank_(x1,x2):
    """
    Griewank function for a given x1 and x2.
    
    arguments:
        x1 (float): the first value
        x2 (float): the second value

    returns:
         value of the function
    """

    A = x1**2/4000 + x2**2/4000
    B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))
    return A-B+1

def griewank_minimizer(brackets = [-600,600], tau = 10**(-8), warm_up_K = 10, K = 1000, seed = 2023, do_print = False):
    """
    The refined global minimizer of the Griewank function using a multi-start method.

    arguments:
        brackets (array): the brackets for the uniform distribution
        tau (float): the threshold for the stopping criterion
        warm_up_K (int): the number of iterations before the algorithm starts updating x
        K (int): the number of iterations
        seed (int): the seed for the random number generator
        do_print (bool): whether to print the result

    returns:
        dict: a dictionary with the solution, the number of iterations and the value of the function
    """

    np.random.seed(seed)
    # a. draw random points from the uniform distribution
    x = np.random.uniform(brackets[0],brackets[1],(K,2))

    # set number of iterations to 0
    k = 0
    while k < K:
        # b. control if still warming up
        if k >= warm_up_K:
            # c. calculate chi_k 
            chi_k = 0.5 * (2 / (1 + np.exp((k - warm_up_K) / 100)))
            # d. update x
            x[k] = chi_k*x[k] + (1-chi_k)*x_star
        # e. minimize using the scipy.optimize.minimize with BFGS method
        x_opt = optimize.minimize(fun=griewank, x0=x[k], method = 'BFGS', tol = tau).x
        # f. set optimal value as x_star
        if k == 0:
            x_star = x_opt
        elif griewank(x_opt) < griewank(x_star):
            x_star = x_opt
        # update k
        k += 1
        # g. check if x_star is below threshold
        if griewank(x_star) < tau:
            value = griewank(x_star)
            if do_print:
                print('The global minimum is approximately: f = {value:.2e}\n' 
                      f'The solution is x_1 = {x_star[0]:.2e} and x_2 = {x_star[1]:.2e} \n'
                      f'Iterations: {k}')
            # 4. return x_star as x_star and k as number of iterations as a dictionary
            return {'sol':x_star, 'iter':k, 'value':value}