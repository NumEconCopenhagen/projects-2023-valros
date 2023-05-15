from types import SimpleNamespace
import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt

class SolowModelClass:
    def __init__(self):

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

    def setup(self):
        """ Initialize the class with default values. """

        par = self.par

        # a. Model parameters
        par.alpha = 1/3
        par.epsilon = 1/6
        par.delta = 0.05
        par.s_Y = 0.1
        par.s_E = 0.005
        par.n = 0.01
        par.g = 0.02

        # b. Initial values
        par.K0 = 1 # initial capital
        par.R0 = 1 # initial amount of limited resource
        par.L0 = 1 # initial labor
        par.A0 = 1 # initial technology

        # c. simulation parameters
        par.T = 100 # number of periods
        

    def analytic_ss(do_print=False):
        
        # a. set up sympy symbols
        zstar = sm.symbols('z')
        alpha = sm.symbols('alpha')
        epsilon = sm.symbols('epsilon')
        delta = sm.symbols('delta')
        s_Y = sm.symbols('s_Y')
        s_E = sm.symbols('s_E')
        g = sm.symbols('g')
        n = sm.symbols('n')        
        # b. define equation for ss
        denom = (((1 + g) * (1 + n))**(1-epsilon-alpha) * (1-s_E)**epsilon)**(1/(1-alpha))
        ss = sm.Eq(zstar,(1/denom)* (s_Y + (1 - delta) * zstar))

        # c. solve for ss
        zss = sm.solve(ss,zstar)[0]

        # d. print ss
        if do_print==True:
            print(sm.Eq(zstar,zss))
        # e. return ss
        return zss

    def evaluate_ss(self, ss, do_print=False):
        par = self.par

        # a. set up sympy symbols
        alpha = sm.symbols('alpha')
        epsilon = sm.symbols('epsilon')
        delta = sm.symbols('delta')
        s_Y = sm.symbols('s_Y')
        s_E = sm.symbols('s_E')
        g = sm.symbols('g')
        n = sm.symbols('n')   

        # a. lambdify zss
        eq = sm.lambdify((alpha,epsilon,delta,s_Y,s_E,g,n),ss)

        # b. evaluate
        sol = eq(par.alpha,par.epsilon,par.delta,par.s_Y,par.s_E,par.g,par.n)

        # c. print
        if do_print==True:
            print(f'z = {sol}')
        
        # d. return
        return sol

    def solve_ss(self,method='bisect', do_print=False):
        """ Solve for the steady state of the model. """
        
        par = self.par

        # a. objective function
        obj = lambda z: (1 / (((1+par.g)*(1+par.n))**(1-par.epsilon-par.alpha)*(1-par.s_E)**(par.epsilon))) * (par.s_Y + (1-par.delta)*z)**(1-par.alpha) * z**par.alpha - z

        #. b. call root finder
        if method == 'bisect' or method == 'brentq':
            result = optimize.root_scalar(obj,bracket=[0.1,100],method=method)
        else:
            raise ValueError('method must be bisect, or brentq')
        
        # c. print result
        if do_print == True:
            print(result)
        
        return result
    
    def simulate(self,do_print=False):
        par = self.par
        sim = self.sim

        # a. initialize
        sim.K = np.empty(par.T+1)
        sim.R = np.empty(par.T+1)
        sim.L = np.empty(par.T+1)
        sim.A = np.empty(par.T+1)
        sim.Y = np.empty(par.T+1)
        sim.E = np.empty(par.T+1)
        sim.z = np.empty(par.T+1)
        sim.t = np.linspace(0,par.T+1,par.T+1)

        # b. initial values
        sim.K[0] = par.K0
        sim.R[0] = par.R0
        sim.L[0] = par.L0
        sim.A[0] = par.A0
        sim.E[0] = par.s_E * sim.R[0]
        sim.Y[0] = sim.K[0]**par.alpha * (sim.A[0] * sim.L[0])**(1-par.alpha-par.epsilon) * sim.E[0]**par.epsilon
        sim.z[0] = sim.K[0] / sim.Y[0]

        # c. simulate
        for t in range(par.T):
            sim.K[t+1] = (1-par.delta) * sim.K[t] + par.s_Y * sim.Y[t]
            sim.R[t+1] = (1-par.s_E) * sim.R[t]
            sim.L[t+1] = (1+par.n) * sim.L[t]
            sim.A[t+1] = (1+par.g) * sim.A[t]
            sim.E[t+1] = par.s_E * sim.R[t+1]
            sim.Y[t+1] = sim.K[t+1]**par.alpha * (sim.A[t+1] * sim.L[t+1])**(1-par.alpha-par.epsilon) * sim.E[t+1]**par.epsilon
            sim.z[t+1] = sim.K[t+1] / sim.Y[t+1]
        
        # d. plot
        if do_print == True:
            fig, ax = plt.subplots(2,3)
            ax[0,0].plot(sim.t,sim.K,label='$K_t$')
            ax[0,0].set_title('Capital')
            ax[1,0].plot(sim.t,sim.Y,label='$Y_t$')
            ax[1,0].set_title('Output')
            ax[0,2].plot(sim.t,sim.R,label='$R_t$')
            ax[0,2].set_title('Limited ressource')
            ax[0,1].plot(sim.t,sim.L,label='$L_t$')
            ax[0,1].set_title('Labor')
            ax[1,1].plot(sim.t,sim.A,label='$A_t$')
            ax[1,1].set_title('Technology')
            ax[1,2].plot(sim.t,sim.E,label='$E_t$')
            ax[1,2].set_title('Consumption of limited ressource')
            plt.show()
    
    def convergence_plot(self):
        par = self.par

        # a. simulate
        z = np.linspace(0,2,250)
        z_1 = (1/((1+par.g)*(1+par.n)**(1-par.alpha-par.epsilon) * (1-par.s_E)**(par.epsilon))) * (par.s_Y + (1-par.delta)*z)**(1-par.alpha) * z**(par.alpha)
        
        linex = np.linspace(0,2,250)
        liney = linex

        # a. plot z[t+1] against z[t]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(z,z_1)
        ax.plot(linex,liney,'--', color='black')
        ax.set_xlabel('$z_t$')
        ax.set_ylabel('$z_{t+1}$')
        ax.set_title('Convergence plot')
        plt.show()