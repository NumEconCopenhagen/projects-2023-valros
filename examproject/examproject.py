import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import sympy as sm
from IPython.display import display
from types import SimpleNamespace

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

    def labour_fun(self):
        """
        Calculates the labour supply.

        arguments:
            none

        returns:
            labour supply
        """

        # a. call parmeter values and necessary symbols
        par = self.par
        alpha = sm.symbols('alpha', positive=True)
        kappa = sm.symbols('kappa', positive=True)
        w = sm.symbols('w', positive=True)
        nu = sm.symbols('nu', positive=True)
        tau = sm.symbols('tau', positive=True)

        # b. call analytical_sol function and extract solution as a function of w and tau
        sol = analytical_sol()[1]

        # c. define function for labour supply
        L_func = sm.lambdify((alpha, nu, kappa, w, tau), sol)

        # d. calculate labour supply and return
        return L_func(alpha=par.alpha, kappa=par.kappa, nu=par.nu, w=par.w, tau=par.tau)
    
    def govern_fun(self):
        """
        Calculates the government spending.

        arguments:
            none

        returns:
            government spending
        """

        # a. call parmeter values
        par = self.par

        # b. calculate government spending and return
        return par.alpha * par.w * self.labour_fun()
    
    def consump_fun(self):
        """
        Calculates the consumption.

        arguments:
            none

        returns:
            consumption
        """

        # a. call parmeter values
        par = self.par

        # b. calculate consumption and return
        return par.kappa + (1-par.tau) * par.w * self.labour_fun(alpha=par.alpha, nu=par.nu, kappa=par.kappa, w=par.w, tau=par.tau)
    
    def util_fun(self):
        """
        Calculates the utility.

        arguments:
            none

        returns:
            utility
        """

        # a. call parmeter values
        par = self.par

        # b. calculate utility and return
        return np.log(self.consump_fun(w=par.w, tau=par.tau, kappa=par.kappa, alpha=par.alpha, nu=par.nu)**par.alpha * self.govern_fun(w=par.w, tau=par.tau, alpha=par.alpha, nu=par.nu, kappa=par.kappa)**(1-par.alpha)) - par.nu * (self.labour_fun(alpha=par.alpha, nu=par.nu, kappa=par.kappa, w=par.w, tau=par.tau)**2)/2
        
    def tax_plot(self):

        # a. call parmeter values
        par = self.par

        # b. define array of tau values
        par.tau = np.linspace(0, 1, 100)

        # c. call labour, government spending, and utility functions
        labour = self.labour_fun()
        gover = self.govern_fun()
        util = self.util_fun()

        # d. plot labour supply, government spending, and utility as a function of tau
        fig = plt.figure()
        fig.suptitle('Labor Supply, Government Consumption, and Worker Utility as Functions of the Tax Rate', fontsize = 15)

        # create subplots
        plt.subplot(3, 1, 1)
        plt.plot(par.tau, labour, label='Labor Supply (L)')
        #plt.xlabel('Tax Rate ' +r'$\tau$', fontsize = 10)
        plt.ylabel(r'Labor Supply (L)', size = 10)
        plt.legend(fontsize = 10, loc = 'lower center')

        plt.subplot(3, 1, 2)
        plt.plot(tau_vec, gov_vec, label='Government Consumption (G)', color = 'red')
        #plt.xlabel('Tax Rate ' +r'$\tau$', fontsize = 10)
        plt.ylabel(r'Government Consumption (G)', size = 10)
        plt.legend(fontsize = 10, loc = 'lower center')

        plt.subplot(3, 1, 3)
        plt.plot(tau_vec, util_vec, label='Worker Utility (V)', color = 'green')
        plt.xlabel('Tax Rate ' +r'$\tau$', fontsize = 10)
        plt.ylabel(r'Worker Utility (V)', size = 10)
        plt.legend(fontsize = 10, loc = 'lower center')

        plt.tight_layout()
        plt.show()


# functions for problem 3
def griewank(x):
    """
    Returns the value of the Griewank function for a given array x.

    arguments:
        x (array): the array for which to calculate the value of the function

    returns:
        float: the value of the function
    """

    return griewank_(x[0],x[1])
    
def griewank_(x1,x2):
    """
    Griewank function for a given x1 and x2.
    
    arguments:
        x1 (float): the first value
        x2 (float): the second value

    returns:
        float: the value of the function
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
        x_opt = optimize.minimize(fun=griewank, x0=x[k], method = 'BFGS').x
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
                print(f'The global minimum is approximately: f = {value:.2e}\n' 
                      f'The solution is x_1 = {x_star[0]:.2e} and x_2 = {x_star[1]:.2e} \n'
                      f'Iterations: {k}')
            # 4. return x_star as x_star and k as number of iterations as a dictionary
            return {'sol':x_star, 'iter':k, 'value':value}

