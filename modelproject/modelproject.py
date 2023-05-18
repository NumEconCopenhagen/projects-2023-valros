# import standard modules
import numpy as np
from scipy import optimize
import sympy as sm
from types import SimpleNamespace

# import plotting modules
from IPython.display import display
import matplotlib.pyplot as plt

def analytic_ss(ext = False, do_print = False):
    """
    Uses sympy to solve for the steady state of the model.

    arguments:
        ext = include our extension or not (default = False)
        do_print = True or False (default = False)

    returns:
        zss = analytical steady state of z
    """
    # a. set up sympy symbols
    zstar = sm.symbols('z')
    alpha = sm.symbols('alpha')
    delta = sm.symbols('delta')
    s_Y = sm.symbols('s_Y')
    s_E = sm.symbols('s_E')
    g = sm.symbols('g')
    n = sm.symbols('n')

    if ext == True:
        epsilon = sm.symbols('epsilon')
    else:
        epsilon = 0

    # b. define equation for ss
    denom = (((1 + g) * (1 + n))**(1-epsilon-alpha) * (1-s_E)**epsilon)**(1/(1-alpha))
    ss = sm.Eq(zstar,(1/denom)* (s_Y + (1 - delta) * zstar))

    # c. solve for ss
    zss = sm.solve(ss,zstar)[0]

    # d. print ss
    if do_print == True:
        if ext == True:
            print('The analytical steady state of the extended model is:')
        else:
            print('The analytical steady state of the baseline model is:')
        display(sm.Eq(zstar,zss))
    
    # e. return ss
    return zss

class SolowModelClass:
    def __init__(self):
        """ 
        Initialize class and create empty simple namespace.

        arguments:
            none

        returns:
            none 
        """

        # a. create empty simple namespace
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

    def setup(self):
        """
        Define parameter values
        
        arguments:
            ext = include extended model (default = False)
        
        returns:
            none
        """

        # a. load simple namespace
        par = self.par

        # b. set model parameters
        par.alpha = 1/3
        par.epsilon = 1/6
        par.delta = 0.05
        par.s_Y = 0.1
        par.s_E = 0.005
        par.n = 0.01
        par.g = 0.02
        
        # c. set initial values
        par.K0 = 1 # initial capital
        par.R0 = 200 # initial amount of limited resource
        par.L0 = 1 # initial labor
        par.A0 = 1 # initial technology

    def evaluate_ss(self, ss, ext = False, do_print=False):
        """
        Evaluates the analytical steady state of the model.

        arguments:
            ss = analytical steady state of model
            do_print = True or False (default = False)

        returns:
            sol = steady state value of model
            if do_print = True, then also prints the steady state value
        """

        # a. load parameters
        par = self.par

        # b. adjust parameters if extended model
        if ext == True:
            eps = par.epsilon
        else:
            eps = 0

        # c. set up sympy symbols
        alpha = sm.symbols('alpha')
        epsilon = sm.symbols('epsilon')
        delta = sm.symbols('delta')
        s_Y = sm.symbols('s_Y')
        s_E = sm.symbols('s_E')
        g = sm.symbols('g')
        n = sm.symbols('n')   

        # d. lambdify zss
        eq = sm.lambdify((alpha,epsilon,delta,s_Y,s_E,g,n),ss)

        # e. evaluate
        sol = eq(par.alpha,eps,par.delta,par.s_Y,par.s_E,par.g,par.n)

        # f. print
        if do_print==True:
            print(f'z = {sol}')
        
        # g. return
        return sol

    def solve_ss(self,method='brentq', ext = False, do_print=False):
        """ 
        Solve for the steady state of the model. 
        arguments: 
            method = 'bisect' or 'brentq' (default = 'brentq')
            do_print = True or False (default = False)
        
        returns:
            result = steady state value of z
            if do_print = True, then also prints the steady state value
        """
        
        # a. load parameters
        par = self.par

        # b. adjust parameters if extended model
        if ext == True:
            eps = par.epsilon
        else:
            eps = 0

        # c. objective function
        obj = lambda z: (1 / (((1+par.g)*(1+par.n))**(1-eps-par.alpha)*(1-par.s_E)**(eps))) * (par.s_Y + (1-par.delta)*z)**(1-par.alpha) * z**par.alpha - z

        # d. call root finder
        if method == 'bisect' or method == 'brentq':
            result = optimize.root_scalar(obj,bracket=[0.1,100],method=method)
        else:
            raise ValueError('method must be bisect, or brentq')
        
        # e. print result
        if do_print == True:
            print(result)
        
        # f. return result
        return result
    
    def simulate(self, periods = 100, ext=False, do_print=False):
        """
        Simulates the model.

        arguments:
            do_print = True or False (default = False)
            periods = number of periods to simulate (default = 100)
            ext = include extended model (default = False)

        returns:
            if do_print = True, then plots the simulated model for K, Y, L, A, E and R
        """
        # a. load parameters
        par = self.par
        sim = self.sim
        T = periods

        # b. adjust parameters if extended model
        if ext == True:
            eps = par.epsilon
        else:
            eps = 0

        # c. create empty arrays to store simulation
        sim.K = np.empty(T+1)
        sim.R = np.empty(T+1)
        sim.L = np.empty(T+1)
        sim.A = np.empty(T+1)
        sim.Y = np.empty(T+1)
        sim.E = np.empty(T+1)
        sim.z = np.empty(T+1)
        sim.t = np.linspace(0,T+1,T+1)

        # d. initial values
        sim.K[0] = par.K0
        sim.R[0] = par.R0
        sim.L[0] = par.L0
        sim.A[0] = par.A0
        sim.E[0] = par.s_E * sim.R[0]
        sim.Y[0] = sim.K[0]**par.alpha * (sim.A[0] * sim.L[0])**(1-par.alpha-eps) * sim.E[0]**eps
        sim.z[0] = sim.K[0] / sim.Y[0]

        # e. simulate
        for t in range(T):
            sim.K[t+1] = (1-par.delta) * sim.K[t] + par.s_Y * sim.Y[t]
            sim.R[t+1] = (1-par.s_E) * sim.R[t]
            sim.L[t+1] = (1+par.n) * sim.L[t]
            sim.A[t+1] = (1+par.g) * sim.A[t]
            sim.E[t+1] = par.s_E * sim.R[t+1]
            sim.Y[t+1] = sim.K[t+1]**par.alpha * (sim.A[t+1] * sim.L[t+1])**(1-par.alpha-eps) * sim.E[t+1]**eps
            sim.z[t+1] = sim.K[t+1] / sim.Y[t+1]
        
        # f. plot
        if do_print == True:
            if ext == True: # adding plot of limited ressource
                fig, ax = plt.subplots(2,3)
                fig.suptitle('Simulated model with limited ressource', size = 20)
                ax[0,2].plot(sim.t,sim.R)
                ax[0,2].set_title('Limited ressource, $R_t$')
                ax[1,2].plot(sim.t,sim.E)
                ax[1,2].set_title('Consumption of limited ressource, $E_t$')
            else:
                fig, ax = plt.subplots(2,2)
                fig.suptitle('Simulated model', size = 20)
            ax[0,0].plot(sim.t,sim.K)
            ax[0,0].set_title('Capital stock, $K_t$')
            ax[1,0].plot(sim.t,sim.Y)
            ax[1,0].set_title('Output, $Y_t$')
            ax[0,1].plot(sim.t,sim.L)
            ax[0,1].set_title('Labor, $L_t$')
            ax[1,1].plot(sim.t,sim.A)
            ax[1,1].set_title('Technology, $A_t$')
            plt.show()
    
    def convergence_plot(self, ext=False):
        """
        Plots the phase diagram for the model.

        arguments:
            ext = include extended model (default = False)

        returns:
            plot of the phase diagram
        """

        # a. load parameters
        par = self.par

        eps = par.epsilon
        # b. adjust parameters if extended model
        #if ext == True:
        #    eps = par.epsilon
        #else:
        #    eps = 0

        # c. simulate
        z = np.linspace(0,1.75,1000)
        z_1 = (1/(((1+par.g)*(1+par.n))**(1-par.alpha-eps) * (1-par.s_E)**(eps))) * (par.s_Y + (1-par.delta)*z)**(1-par.alpha) * z**(par.alpha)
        
        # d. create 45 degree line
        linex = np.linspace(0,1.75,2)
        liney = linex

        # e. add line for steady state
        ss = self.solve_ss(ext=ext).root

        # f. plot z[t+1] against z[t]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(z,z_1, label = f'$z_{{t+1}}$')
        ax.plot(linex,liney,'--', color='black', label = '45 degree line')
        ax.set_xlabel('$z_t$')
        ax.set_ylabel('$z_{t+1}$')
        ax.plot(ss,ss,'o', color='black', label=f'Steady state {ss:.4f}')
        ax.set_title('Convergence plot')
        plt.axhline(y=ss, color='black')
        plt.axvline(x=ss, color='black')
        plt.legend()
        plt.show()