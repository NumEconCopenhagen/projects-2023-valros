import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import sympy as sm
from IPython.display import display
from types import SimpleNamespace

from IPython.core.display import HTML

HTML("""
<style>
 {
    display: table-cell;
    text-align: center;
   .output_png vertical-align: middle;
}
</style>
""")


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
        tau = np.linspace(0, 1, 500)

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
            tau = np.linspace(0, 1, 500)

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

