# FD and CD is unstable (explicit) for stiff system > need to use small timestep
# Backward euler is more stable; implicit method of order 1
# Trapezoidal; implicit method of order 2
# Use sparse matrix solve scipy
# Use RK-scheme to solve equation simulated curve
# Soln depends on n
# works rather good and fast even with small N
# Find critical point in calculation: If critic_num < tol :
    # skip all other steps and set to zero or something
    # or calculate Y_nl only for values above tol and set zero otherwise
    # find point at which Y_nl would be zero and dont 
    # calculate values further from that point
    # try crank-nicholson scheme =>  this yields better results, especially for n
    # Newtons method gets expensive for large conversion values: try approximating Y**n with lagrange polynomial
    # the lagrange polynomial has top be included in the Y_matrix: i.e. the tridiagonal matrix for the derivative expression
    # partial components: dx/dt = A1*f(x1)+A2*f(x2) or A*f(x1+x2)?
    # we know that xt = c1*x1 + c2*x2
    # Check that rigth T_exp indices are used. Can have error of 1.2 C if not!
# sparse faster for larger matrices >200

# 500:
#Total minimization time    = 7.16 seconds
#Nonlinear equation solving = 6.74 seconds

#Total minimization time    = 25.55 seconds
#Nonlinear equation solving = 23.95 seconds

import numpy as np
from numpy import ones,diag
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize,basinhopping,newton
from numpy.linalg import norm,solve
import time
from scipy.sparse import diags,csc_matrix
from scipy.sparse.linalg import spsolve


T_start = 620
T_end = 880

#number of data points parameter estimation
N=200


# simulated experimental curve
A_ex = 10**8.4
E_ex = 145000
n_ex= 0.5


T_exp = np.linspace(T_start,T_end,N+1)

T_int = np.linspace(T_start,T_end,1000)

dt = (T_end-T_start)/N



def tridiag(v, d, w, N,p):
    '''
    Help function 
    Returns a tridiagonal matrix A=tridiag(v, d, w) of dimension N x N.
    '''
    
    e = ones(N)       # array [1,1,...,1] of length N
    A = v*diag(e[1:],-1)+d*diag(1+((T_end-T_start)/N)*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T_exp[1:])))
    return A

def jac(y,p):
    y = y.clip(1e-16)
    '''
    Help function 
    Returns a tridiagonal matrix A=tridiag(v, d, w) of dimension N x N.
    '''
    
    e = ones(N)       # array [1,1,...,1] of length N
    A = -diag(e[1:],-1)+diag(1+(T_end-T_start)/N*(10**(p[0]))*(p[2]/100)*np.exp(-(p[1]*1000)/(8.315*T_exp[1:]))*y[:]**(p[2]/100-1))
    A[0][0]=1
    return A

def newton_system_trap(func, jac, x0,p, tol = 1.e-3, max_iter=30):
    x = x0
    for k in range(max_iter):
        fx = func(x,p)
        if norm(fx, np.inf) < tol:          # The solution is accepted. 
            break
        Jx = jac(x,p)
        delta = solve(Jx, -fx) 
        x = x + delta 
        x = x.clip(0) #if x<0 replace by 0 to avoid numerical issues
    return x

def newton_system_sparse(func, jac, x0,p, tol = 1.e-3, max_iter=20):
    x = x0
    for k in range(max_iter):
        fx = func(x,p)
        if norm(fx, np.inf) < tol:          # The solution is accepted. 
            break
        Jx = jac(x,p)
        delta = spsolve(Jx, -fx) 
        x = x + delta 
        x = x.clip(0) #if x<0 replace by 0 to avoid numerical issues
    return x

def func(x,p):
    #x = x.clip(1e-12) # add if errors 
    y = np.array([-x[0:-1]+x[1:]+(T_end-T_start)/N*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T_exp[2:]))*(x[1:])**(p[2]/100)])
    y = np.insert(y,0,0) #y_0 - 1 = 1 - 1 = 0
    return y

def func_trap(x,p):
    #x = x.clip(1e-12) # add if errors 
    y = np.array([-x[0:-1]+x[1:]+0.5*(T_end-T_start)/N*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T_exp[2:]))*(x[1:])**(p[2]/100)+ 0.5*(T_end-T_start)/N*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T_exp[2:]))*(x[0:-1])**(p[2]/100)])
    y = np.insert(y,0,0) #y_0 - 1 = 1 - 1 = 0
    return y

def jac_trap(y,p):
    y = y.clip(1e-16)
    '''
    Help function 
    Returns a tridiagonal matrix A=tridiag(v, d, w) of dimension N x N.
    '''
    
    e = ones(N)       # array [1,1,...,1] of length N
    A = diag(-1+e[1:]*0.5*(T_end-T_start)/N*(10**(p[0]))*(p[2]/100)*np.exp(-(p[1]*1000)/(8.315*T_exp[1:-1]))*y[:-1]**(p[2]/100-1),-1)+diag(1+0.5*(T_end-T_start)/N*(10**(p[0]))*(p[2]/100)*np.exp(-(p[1]*1000)/(8.315*T_exp[1:]))*y[:]**(p[2]/100-1))
    A[0][0]=1
    return A

def func_trap_sparse(x,p):
    #x = x.clip(1e-12) # add if errors 
    y = np.array([-x[0:-1]+x[1:]+0.5*(T_end-T_start)/N*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T_exp[2:]))*(x[1:])**(p[2]/100)+ 0.5*(T_end-T_start)/N*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T_exp[2:]))*(x[0:-1])**(p[2]/100)])
    y = np.insert(y,0,0) #y_0 - 1 = 1 - 1 = 0
    return y

def jac_trap_sparse(y,p):
    y = y.clip(1e-16)
    '''
    Help function 
    Returns a tridiagonal matrix A=tridiag(v, d, w) of dimension N x N.
    '''
    
    e = ones(N)       # array [1,1,...,1] of length N
    di = 1+0.5*(T_end-T_start)/N*(10**(p[0]))*(p[2]/100)*np.exp(-(p[1]*1000)/(8.315*T_exp[1:]))*y[:]**(p[2]/100-1)
    di[0] = 1
    A = diags(-1+e[1:]*0.5*(T_end-T_start)/N*(10**(p[0]))*(p[2]/100)*np.exp(-(p[1]*1000)/(8.315*T_exp[1:-1]))*y[:-1]**(p[2]/100-1),-1)+diags(di)
    return A
    #return csc_matrix(A)

def sparse_tridiag(N,p):
    e = ones(N)
    di = e*(1+((T_end-T_start)/N)*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T_exp[1:])))
    di[0]=1
    return diags(-e[1:],-1)+diags(di)

def of_sparse(p):
    A_m_=sparse_tridiag(N,p)
    
    Y_ = spsolve(A_m_,b)
    #print(Y_)
    #Y_nl = newton_system(func,jac,Y_,p)
    
    t0_py = time.time()
    Y_sparse = newton_system_sparse(func_trap_sparse,jac_trap_sparse,Y_,p)
    t1_py = time.time()
    times_nonlinear.append(t1_py-t0_py)
    
    #return np.sum((1-Y_trap-y_exp[1:])**2)
    return np.sum(((Y_sparse[:-1]-Y_sparse[1:])/dt-dx[2:])**2)


def fx(T,y):
    return A_ex*np.exp(-E_ex/(8.315*T))*(1-y)**n_ex 

sol = solve_ivp(fx,(T_start,T_end),[0],dense_output=True,max_step=0.5)
y_exp = 0.3*sol.sol(T_exp)[0]
dx = A_ex*np.exp(-E_ex/(8.315*T_exp))*(1-sol.sol(T_exp)[0])**n_ex

am,em,nm = 1,1.04,2
def fx2(T,y):
    return A_ex*am*np.exp(-E_ex*em/(8.315*T))*(1-y)**(n_ex*nm)

sol2 = solve_ivp(fx2,(T_start,T_end),[0],dense_output=True,max_step=0.5)
y_exp2 = 0.7*sol2.sol(T_exp)[0]
dx2 = A_ex*am*np.exp(-E_ex*em/(8.315*T_exp))*(1-sol2.sol(T_exp)[0])**(n_ex*nm)


dx_exp = (dx+dx2)
b=np.zeros(N)
b[0] = 1 # initial condition: y0 = 1


def of(p):
    A_m_=tridiag(-1,1,0,N,p[:3])
    A_m_[0][0]=1
    
    Y_ = np.linalg.solve(A_m_,b)
    #print(Y_)
    #Y_nl = newton_system(func,jac,Y_,p)
    
    t0_py = time.time()
    Y_trap = newton_system_trap(func_trap,jac_trap,Y_,p[:3])
    t1_py = time.time()
    times_nonlinear.append(t1_py-t0_py)
    
    
    A_m_2=tridiag(-1,1,0,N,[p[0],p[3],p[4]])
    A_m_2[0][0]=1
    Y_2 = np.linalg.solve(A_m_2,b)
    Y_trap2 = newton_system_trap(func_trap,jac_trap,Y_2,[p[0],p[3],p[4]])
    #dx_calc = (Y_trap[:-1]-Y_trap[1:])/dt
    dx_calc = (Y_trap[:-1]-Y_trap[1:])/dt + (Y_trap2[:-1]-Y_trap2[1:])/dt
    #return np.sum(((1-Y_trap)+(1-Y_trap2)+(-y_exp[1:]-y_exp2[1:]))**2)
    return np.sum((+dx_calc-dx_exp[2:])**2)





times_nonlinear =[]
times_min = []

t0 = time.time()
res = minimize(of,[8.6,150,100,150,50],tol=1e-8,method='Nelder-mead',options={'maxiter':2000,'xatol':1e-8,'fatol':1e-8,'maxfev':2000})
t1=time.time()
times_min.append(t1-t0)
#res=basinhopping(of, [8,150,100,9,200,100],niter=200,stepsize=1)




A_m=tridiag(-1,1,0,N,res.x[:3])
A_m[0][0]=1
Y = np.linalg.solve(A_m,b)
#Y_nl = newton_system(func,jac,Y,res.x)
Y_trap = newton_system_trap(func_trap, jac_trap, Y, res.x[:3])


#A_m_s=sparse_tridiag(N,res.x)
#Y_sparse= spsolve(A_m_s,b)
#Y_nl = newton_system(func,jac,Y,res.x)
#Y_sparse = newton_system_sparse(func_trap_sparse, jac_trap_sparse, Y_sparse, res.x)

A_m_2=tridiag(-1,1,0,N,[res.x[0],res.x[3],res.x[4]])
A_m_2[0][0]=1
Y_2 = np.linalg.solve(A_m_2,b)
Y_trap2 = newton_system_trap(func_trap,jac_trap,Y_2,[res.x[0],res.x[3],res.x[4]])


def scale(c):
    return np.sum(((c[0]*Y_trap+(1-c[0])*(Y_trap2))-(y_exp[1:]+y_exp2[1:]))**2)

c = minimize(scale,100,method='Powell')


f,(a,a2) = plt.subplots(2,1,sharex=True)
#a.plot(T_exp[1:],1-Y_nl,label='backward euler',ls='--')
a.plot(T_exp[1:],1-Y_trap[:],label='trapezoidal rule')
a.plot(T_exp[1:],1-Y_trap2[:],label='trapezoidal rule')
a.plot(T_exp,(y_exp+y_exp2),label='simulated',ls=':',lw=1.5) #simulated
#a.plot(T_exp[1:],2-Y_trap-Y_trap2)
a.plot(T_exp[1:],1-c.x[0]*Y_trap-(1-c.x[0])*Y_trap2)
#a.plot(T_exp[1:],1-Y_sparse[:],label='trapezoidal rule')
a.set(ylabel='conversion')


dx_calc1 = N/(T_end-T_start)*(Y_trap[:-1]-Y_trap[1:])
dx_calc2 = N/(T_end-T_start)*(Y_trap2[:-1]-Y_trap2[1:])

#a2.plot(T_exp[2:],N/(T_end-T_start)*(Y_nl[:-1]-Y_nl[1:]),ls='--') # (f(x)-f(x-h))/h = df/dt
a2.plot(T_exp[2:],dx_calc1)
a2.plot(T_exp[2:],dx_calc2)
a2.plot(T_exp[2:],dx_calc1+dx_calc2)
#a2.plot(T_exp[2:],N/(T_end-T_start)*(Y_sparse[:-1]-Y_sparse[1:]))
#a2.plot(T_exp,dx_calc)
a2.plot(T_exp,dx_exp,lw=1.5,ls=':') # simulated

a2.set(xlabel='Temperature (Kelvin)',ylabel='conversion rate')

#a2.plot(datalist[k]['t1'].iloc[2:],1/60*(Y[:-1]-Y[1:])/(T_end-T_start)*N,lw=0.8)
a.legend()
#f.show()
k=19
print("#"*k + "\nSimulated data:\n" + "#"*k)
print('log10(A)= {:.3f}   E= {:.3f}   n={:.3f}   log10(A2)= {:.3f}   E2= {:.3f}   n2={:.3f}'.format(np.log10(A_ex),E_ex/1000,n_ex,np.log10(A_ex*am),E_ex*em/1000,n_ex*nm))
print('\n'+'#'*k+"\nExperimental data:\n"+"#"*k)
print('log10(A)= {:.3f}   E1= {:.3f}   n1={:.3f}  \t\t\t\t\t E2= {:.3f}  n2= {:.3f}'.format(res.x[0],res.x[1],res.x[2]/100,res.x[3],res.x[4]/100))
print('\n\n'+'Total minimization time    = {:.2f} seconds'.format(sum(times_min)))
print('Nonlinear equation solving = {:.2f} seconds'.format(sum(times_nonlinear)))
