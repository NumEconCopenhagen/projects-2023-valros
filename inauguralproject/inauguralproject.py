#Defining functions used in utility function

#Defining the function C
def C(Lm,Lf,wm=1,wf=1):
    return Lm*wm+Lf*wf

#Definition of the function H
def H(Hm,Hf,alpha=0.5,sigma=1):
    if sigma == 0:
        return min(Hm,Hf)
    elif sigma == 1:
        return Hm**(1-alpha)*Hf**alpha
    else:
        return ((1-alpha)Hm**((sigma-1)/sigma)+alpha*Hf**((sigma-1)/sigma))**(sigma/(sigma-1))

#Definition of the function Q
def Q(C,H,omega=0.5):
    return C**omega*H**(1-omega)

#Definition of the function Tm
def Tm(Lm,Hm):
    return Lm+Hm

#Definition of the function Tf
def Tf(Lf,Hf):
    return Lf+Hf


#Defining the utility function
def util(Q,Tm,Tf,rho=2, nu=0.001,epsilon=1):
    return (Q**(1-rho))/(1-rho)-nu((Tm**(1+1/epsilon))/(1+1/epsilon)+(Tf**(1+1/epsilon))/(1+1/epsilon))