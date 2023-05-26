
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

from numba import njit

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        opt = self.opt = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilonM = 1.0
        par.epsilonF = 1.0
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
        opt.residual = np.nan

        # g. extended model
        par.kappa = 1
        par.seed = 1915

    def calc_utility(self,LM,HM,LF,HF):
        """ 
        calculate utility by first finding utility of consumption and then adding disutility of work 
        
        arguments:
            LM: float, male hours of market work
            HM: float, male hours of home work
            LF: float, female hours of market work
            HF: float, female hours of home work
        
        returns:
            utility: utility
        """

        par = self.par
        opt = self.opt

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        with np.errstate(divide='ignore', invalid='ignore'):
            if par.sigma == 1:
                H = HM**(1-par.alpha)*HF**par.alpha
            elif par.sigma == 0:
                H = np.minimum(HM,HF)
            else:
                H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
        
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)

        with np.errstate(invalid='ignore'):
            utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_M = 1+1/par.epsilonM
        epsilon_F = 1+1/par.epsilonF
        TM = LM+HM
        TF = LF+HF

        disutility = par.nu*(TM**epsilon_M/epsilon_M+par.kappa*(TF**epsilon_F)/epsilon_F)
        
        return utility - disutility

    def solve_discrete(self):
        """ 
        solve model discretely, first a meshgrid is created and then the utility is calculated for all combinations.
        
        arguments:
            all parameters are stored in the class and the possible choices are created in the function

        returns:
            opt: solution
        """
        
        par = self.par
        opt = self.opt
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel()
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument and store solution
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        return opt

    def solve_continous(self,basin=False):
        """ 
        solve model continously by using the scipy.optimize package and either the basinhopping or the minimize method.
        model can be solved both using basinhopping and minimize. basinhopping is used when we fear that we might get stuck in a local optimum.

        arguments:
            basin: boolean, if True the basinhopping method is used, if False the minimize method is used
            all other parameters are stored in the class
        
        returns:
            opt: solution
        """

        par = self.par
        opt = self.opt

        # a. defining target function
        def target_function(x,wM,wF):
            if x[0]+x[1] > 24 or x[2]+x[3] > 24:
                return np.inf
            else:
                return -self.calc_utility(x[0],x[1],x[2],x[3]) 

        # b. setting starting value and bounds
        x0=[10,10,10,10] #Initial guess
        bounds = ((0,24),(0,24),(0,24),(0,24)) #Bounds

        # c. solve the model
        if basin: #use basinhopping
            solution = optimize.basinhopping(target_function,
                                                x0,
                                                niter=10, 
                                                minimizer_kwargs={'args': (par.wM,par.wF),'method':'Nelder-Mead','bounds':bounds},
                                                seed = par.seed)
        else: #use minimize
            solution = optimize.minimize(target_function, 
                                         x0,
                                         args= (par.wM,par.wF),
                                         method='Nelder-Mead', 
                                         bounds=bounds
                                         )
        
        # d. store solution
        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]

        return opt

    def solve_wF_vec(self, discrete=False, basin=False, do_print=False):
        """ 
        solve model for different wF using either the discrete or the continous solve functions defined above.

        arguments:
            discrete: boolean, if True the discrete solve function is used, if False the continous solve function is used
            basin: boolean, if True the continous basinhopping method is used, if False the minimize method is used
            do_print: boolean, if True the results are printed to the console

        returns:
            opt: solution
            if do_print: prints results to the console
        """

        par = self.par
        opt = self.opt

        # a. setting up vectors to store results
        opt.logHFHM = np.zeros(par.wF_vec.size)
        opt.HF_vec = np.zeros(par.wF_vec.size)
        opt.HM_vec = np.zeros(par.wF_vec.size)
        opt.LF_vec = np.zeros(par.wF_vec.size)
        opt.LM_vec = np.zeros(par.wF_vec.size)

        # b. loop over wF
        for i,wF in enumerate(par.wF_vec):
            par.wF = wF

            # i. solve with desired method
            if discrete:
                opt = self.solve_discrete()
            elif basin:
                opt = self.solve_continous(basin=True)
            else:
                opt = self.solve_continous()
            
            # ii. store results
            opt.logHFHM[i] = np.log(opt.HF/opt.HM)
            opt.HM_vec[i] = opt.HM
            opt.HF_vec[i] = opt.HF
            opt.LF_vec[i] = opt.LF
            opt.LM_vec[i] = opt.LM


        # c. print
        if do_print:
            for i,wF in enumerate(par.wF_vec):
                print(f'log(wF/wM) = {np.log(par.wF_vec[i]):4.2f}  log(HF/HM) = {opt.logHFHM[i]:4.2f}' + '\n' + 
                      f'    LM = {opt.LM_vec[i]:4.2f}, HM = {opt.HM_vec[i]:4.2f}, LF = {opt.LF_vec[i]:4.2f}, HF = {opt.HF_vec[i]:4.2f}')

        return opt
    
    def run_regression(self):
        """ 
        finding beta0 and beta1 with least squares log(HF/HM) on log(wF/wM)

        arguments:
            all parameters are stored in the class
        
        returns:
            opt: solution
        """

        par = self.par
        opt = self.opt

        x = np.log(par.wF_vec)
        y = np.log(opt.HF_vec/opt.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        opt.beta0,opt.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    

    def estimate(self,mode='normal',do_print=False):
        """ 
        estimating optimal parameter values given the mode. 
        the modes are;
            mode normal is the standard mode, where alpha and sigma is estimated; 
            mode only_sigma is the mode where only sigma is estimated;
            mode extended where epsilonF and sigma is estimated when using the extended model estimation we use basinhopping to find the global optimum.

        arguments:
            mode: string, either 'normal', 'only_sigma' or 'extended'
            do_print: boolean, if True the results are printed to the console

        returns:
            opt: solution
            if do_print: prints results to the console
        """
        
        par = self.par
        opt = self.opt
        
        # a. estimating the model depending on the mode
        if mode == 'normal':

            # i. defining target function
            def target(x):
                par.alpha, par.sigma = x
                self.solve_wF_vec()
                self.run_regression()
                opt.residual = (opt.beta0-par.beta0_target)**2  + (opt.beta1-par.beta1_target)**2
                return opt.residual
            
            # ii. setting starting value and bounds
            x0=[0.5,0.1] # initial guess
            bounds = ((0,1),(0,1)) # bounds

            # iii. solving the model
            solution = optimize.minimize(target,
                                        x0,
                                        method='Nelder-Mead',
                                        bounds=bounds)
            
            # iv. storing the results
            opt.alpha = solution.x[0]
            opt.sigma = solution.x[1]

            # v. printing the results
            if do_print:
                print(f'\u03B1_opt = {opt.alpha:6.4f}') #\u03B1 is the unicode for the greek letter alpha
                print(f'\u03C3_opt = {opt.sigma:6.4f}') #\u03C3 is the unicode for the greek letter sigma
                print(f'Residual_opt = {opt.residual:6.4f}')

        elif mode == 'only_sigma':

            # i. defining target function
            def target(x):
                par.sigma = x
                self.solve_wF_vec()
                self.run_regression()
                opt.residual = (opt.beta0-par.beta0_target)**2  + (opt.beta1-par.beta1_target)**2
                return opt.residual
            
            # ii. setting starting value and bounds
            x0=[0.2] # initial guess
            bounds = ((0,1)) # bounds

            # iii. solving the model
            solution = optimize.minimize_scalar(target,
                                        x0,
                                        method='Bounded',
                                        bounds=bounds)
            
            # iv. storing the results
            opt.sigma = solution.x

            # v. printing the results
            if do_print:
                print(f'\u03C3_base = {opt.sigma:6.4f}') #\u03C3 is the unicode for the greek letter sigma
                print(f'Residual_base = {opt.residual:6.4f}')

        
        elif mode == 'extended':
            
            # i. defining target function for the basinhopping algorithm
            def target(x):
                par.epsilonF, par.sigma = x
                self.solve_wF_vec()
                self.run_regression()
                opt.residual = (opt.beta0-par.beta0_target)**2  + (opt.beta1-par.beta1_target)**2
                return opt.residual
                

            # ii. setting starting value and bounds
            x0=[4.5,0.1] # initial guess
            bounds = ((0,5),(0,1)) # bounds
            
            # iii. solving the model
            solution = optimize.basinhopping(target,
                                        x0,
                                        niter = 20,
                                        minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds},
                                        seed = par.seed)
            
            # iv. storing the results
            opt.epsilonF = solution.x[0]
            opt.sigma = solution.x[1]

            # v. printing the results
            if do_print:
                print(f'\u03B5_F_ext = {opt.epsilonF:6.4f}') #\u03B5 is the unicode for the greek letter epsilon
                print(f'\u03C3_ext = {opt.sigma:6.4f}') #\u03C3 is the unicode for the greek letter sigma
                print(f'Residual_ext = {opt.residual:6.4f}')

        else:
            print('Mode not recognized. Available modes are: normal (default), extended and only_sigma')


### Plot Functions ###

## Question 1

def plot_heat_map(ratio_results, alpha_range, sigma_range):
    '''
    This function plots the results from the heat map.
    
    arguments:
        ratio_results: array, the results from the heat map
        alpha_range: array, the range of alpha values
        sigma_range: array, the range of sigma values
    
    '''
    
    # d. plot results in heatmap
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(ratio_results, cmap='coolwarm')

    # e. create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r'$H_F/H_M$', rotation=270, labelpad=10)

    # f. set title and labels
    plt.title("Ratio between" + " " + r'$H_F$' + " " + "and" + " " r'$H_M$')
    ax.set_xticks(np.arange(len(sigma_range)))
    ax.set_yticks(np.arange(len(alpha_range)))
    ax.set_xticklabels(sigma_range)
    ax.set_yticklabels(alpha_range)
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\alpha$')

    # g. looping over data to create annotations
    for i in range(len(alpha_range)):
        for j in range(len(sigma_range)):
            text = ax.text(j, i, r'$\frac{H_F}{H_M}=$'+f"{ratio_results[i, j]:.2f}",
                        ha="center", va="center", color="black")

    # h. show plot
    plt.show()

## Question 2

def plot_data_points(logwFwM_ratio, logHFHM_ratio):
    '''
    plots a heat map of the log(wF/wM) and log(HF/HM) ratios
    
    arguments:
        logwFwM_ratio: list of log(wF/wM) ratios
        logHFHM_ratio: list of log(HF/HM) ratios
    
    '''
    
    # c. plot results in a plot
    plt.plot(logwFwM_ratio, logHFHM_ratio, 
            marker = 'o', 
            linestyle = '--', 
            label="Estimated discretly")

    # d. set title and labels
    plt.title(r'$\log(\frac{H_F}{H_M})$' + " against " r'$\log(\frac{w_F}{w_M})$' + " solved discretely")
    plt.legend(loc='upper right')
    plt.xlabel(r'$\log(\frac{w_F}{w_M})$')
    plt.ylabel(r'$\log(\frac{H_F}{H_M})$')

    # e. looping over data to create annotations
    for i in range(5):
        plt.annotate(f"({logwFwM_ratio[i]:.2f}, {logHFHM_ratio[i]:.2f})", (logwFwM_ratio[i]+0.005, logHFHM_ratio[i]+0.005))

    # f. show plot
    plt.show()
    
## Question 3

def plot_continious(logwFwM_ratio, logHFHM_ratio_continous):
    '''
    plots a heat map of the log(wF/wM) and log(HF/HM) ratios
    
    arguments:
        logwFwM_ratio: list of log(wF/wM) ratios
        logHFHM_ratio_continous: list of log(HF/HM) ratios solved continously
    
    '''
    
    # c. plot results in a plot
    plt.plot(logwFwM_ratio, logHFHM_ratio_continous, 
            marker = 'o', 
            linestyle = '--', 
            label="Estimated continously")

    # d. set title and labels
    plt.title(r'$\log(\frac{H_F}{H_M})$' + " against " r'$\log(\frac{w_F}{w_M})$' + " solved continously")
    plt.legend(loc='upper right')
    plt.xlabel(r'$\log(\frac{w_F}{w_M})$')
    plt.ylabel(r'$\log(\frac{H_F}{H_M})$')

    # e. looping over data to create annotations
    for i in range(5):
        plt.annotate(f"({logwFwM_ratio[i]:.2f}, {logHFHM_ratio_continous[i]:.2f})", (logwFwM_ratio[i]+0.001, logHFHM_ratio_continous[i]+0.0005))

    # f. show plot
    plt.show()
    

## Question 4

def plot_siminski(logwFwM_ratio, logHFHM_ratio_continous, x, y, model):
    '''
    plots a graph for siminski and yetsenga (2022) and the estimated model
    
    arguments:
        logwFwM_ratio: list of log(wF/wM) ratios
        logHFHM_ratio_continous: list of log(HF/HM) ratios solved continously
        x: list of x values for siminski and yetsenga (2022)
        y: list of y values for siminski and yetsenga (2022)
        model: model object
    
    '''
    
    
    plt.plot(x, y, color = 'red', linestyle = '-', label='Siminski and Yetsenga (2022)')
    plt.plot(logwFwM_ratio, logHFHM_ratio_continous, 
            marker = 'o', 
            linestyle = '--', 
            label = 'Estimated with '+ r'$\alpha=$' f'{model.opt.alpha:.2f}'+ " and " + r'$\sigma=$' f'{model.opt.sigma:.2f}')

    plt.legend(loc='upper right')

    # g. set title and labels
    plt.title(r'$\log(\frac{H_F}{H_M})$' + " against " r'$\log(\frac{w_F}{w_M})$' + " with " + r'$\alpha=$' f'{model.opt.alpha:.2f}'+ " and " + r'$\sigma=$' f'{model.opt.sigma:.2f}')
    plt.xlabel(r'$\log(\frac{w_F}{w_M})$')
    plt.ylabel(r'$\log(\frac{H_F}{H_M})$')

    # h. looping over data to create annotations
    for i in range(5):
        plt.annotate(f"({logwFwM_ratio[i]:.2f}, {logHFHM_ratio_continous[i]:.2f})", (logwFwM_ratio[i]+0.001, logHFHM_ratio_continous[i]+0.0005))

    # i. show plot
    plt.show()
    
## Question 5

def plot_all_three(logwFwM_ratio, logHFHM_ratio_baseline, logHFHM_ratio_extended, x, y, model, extended_model):
    ''''
    plots a graph of the baseline model, the extended model and the Siminski and Yetsenga (2022) data
    
    arguments:
        logwFwM_ratio: list of log(wF/wM) ratios
        logHFHM_ratio_baseline: list of log(HF/HM) ratios solved with the baseline model
        logHFHM_ratio_extended: list of log(HF/HM) ratios solved with the extended model
        x: list of log(wF/wM) ratios
        y: list of log(HF/HM) ratios from Siminski and Yetsenga (2022)
        model: baseline model
        extended_model: extended model
    
    '''
    
    plt.plot(x, y, color='red', linestyle='-', label='Siminski and Yetsenga (2022)')
    plt.plot(logwFwM_ratio, logHFHM_ratio_baseline, 
            marker='o', 
            linestyle='--', 
            label="Baseline Model: " + r'$\sigma=$' f'{model.par.sigma:.2f}')
    plt.plot(logwFwM_ratio, logHFHM_ratio_extended, 
            marker='s', 
            linestyle='--', 
            label="Extended Model: " + r'$\sigma=$' f'{extended_model.par.sigma:.2f}'+ ", "+ r'$\epsilon_F=$' f'{extended_model.par.epsilonF:.2f}')
    plt.legend(loc="best")

    # e. set title and labels
    plt.title(r'$\log(\frac{H_F}{H_M})$' + " against " r'$\log(\frac{w_F}{w_M})$' + " with " + r'$\alpha=$' f'{model.par.alpha:.2f}')
    plt.xlabel(r'$\log(\frac{w_F}{w_M})$')
    plt.ylabel(r'$\log(\frac{H_F}{H_M})$')

    # f. show plot
    plt.show()
