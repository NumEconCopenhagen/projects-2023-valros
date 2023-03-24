
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        opt = self.opt = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        opt.LM_vec = np.zeros(par.wF_vec.size)
        opt.HM_vec = np.zeros(par.wF_vec.size)
        opt.LF_vec = np.zeros(par.wF_vec.size)
        opt.HF_vec = np.zeros(par.wF_vec.size)

        opt.beta0 = np.nan
        opt.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        opt = self.opt

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.minimum(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
        
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        opt = self.opt
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt   

    def solve_continous(self,do_print=False):
        """ solve model continously """
        #Stadig ikke færdig med denne funktion
        par = self.par
        opt = self.opt

        #Target function
        def target_function(x,wM,wF):
            if x[0]+x[1] > 24 or x[2]+x[3] > 24:
                return np.inf
            else:
                return -self.calc_utility(x[0],x[1],x[2],x[3]) 

        #Starting value, bounds and constraints
        x0=[10,10,10,10] #Initial guess
        bounds = ((0,24),(0,24),(0,24),(0,24)) #Bounds

        #Continous solution with the help of the scipy.optimize package
        solution = optimize.minimize(target_function, 
                                     x0,
                                     args= (par.wM,par.wF),
                                     method='Nelder-Mead', 
                                     bounds=bounds
                                     )
        
        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]

        return opt

    def solve_wF_vec(self, discrete=False):
        """ solve model for different wF """

        par = self.par
        opt = self.opt

        logHFHM = np.zeros(par.wF_vec.size)
        optHF = np.zeros(par.wF_vec.size)
        optHM = np.zeros(par.wF_vec.size)
        optLF = np.zeros(par.wF_vec.size)
        optLM = np.zeros(par.wF_vec.size)

        for i,wF in enumerate(par.wF_vec):
            par.wF = wF

            if discrete == True:
                opt = self.solve_discrete()
            else:
                opt = self.solve_continous()

            opt.HM = opt.HM
            opt.HF = opt.HF
            logHFHM[i] = np.log(opt.HF/opt.HM)
            optHM[i] = opt.HM
            optHF[i] = opt.HF
            optLF[i] = opt.LF
            optLM[i] = opt.LM

        opt.logHFHM = logHFHM
        opt.HM_vec = optHM
        opt.HF_vec = optHF
        opt.LF_vec = optLF
        opt.LM_vec = optLM
        return opt
    
    def run_regression(self):
        """ run regression """

        par = self.par
        opt = self.opt

        x = np.log(par.wF_vec)
        y = np.log(opt.HF_vec/opt.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        opt.beta0,opt.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):

        # Unpack the parameters
        alpha, sigma = params
        model.par.alpha = alpha
        model.par.sigma = sigma
        
        # Solve the model for the vector of female wages and run the regression
        model.solve_wF_vec()
        model.run_regression()
        
        beta0_diff = (model.par.beta0_target - model.sol.beta0)**2
        beta1_diff = (model.par.beta1_target - model.sol.beta1)**2
        
        return beta0_diff + beta1_diff

        pass