import pandas as pd
from scipy.interpolate import interp1d,UnivariateSpline
import time




filenames = ['backup/Skrivebord/Kinetic/modulated experiments 2014/R mod 2014-10-15.TXT',
             'backup/Skrivebord/Kinetic/CRR experiments 2014/R ~CRR 2014-10-17.TXT',
             'backup/Skrivebord/Kinetic/20C_min experiments 2014/R 20C_min 2014-10-14.TXT'
             ]
init_mass = []
def read_data():
    for i,filename in enumerate(filenames):
        if i == 0:
            data = pd.read_table(filename,index_col = False,skiprows=38,delim_whitespace = True,names = [ 't1',  'Tbad',  'xbad',  3,  4,  5,  6,  7,  'T1',  'x1', 10, 11, 12, 13, 14, 15, 16])[['t1','T1','x1']]
            init_mass.append(data['x1'][data['T1'] >= 150].iloc[0])
            data = data[data['T1'] >= 640]
            all_data.append(data.set_index(data['t1']*60)) # * 60 for pr sec
            #data = data[data['t1'] <= 273]
            data = data[data['t1'] <= 273]
            data = data[::40]
            datalist.append(data.set_index(data['t1']*60)) 

        elif i ==1:
            data = pd.read_table(filename,index_col = False,skiprows=38,delim_whitespace = True,names = [ 't1',  'T1',  'x1',  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])[['t1','T1','x1']]
            init_mass.append(data['x1'][data['T1'] >= 150].iloc[0])
            data = data[data['T1'] >= 640]
            all_data.append(data.set_index(data['t1']*60))
            data = data[data['t1'] <= 463.5]
            #data = data[data['t1'] <= 302]
            data = data[::40]
            datalist.append(data.set_index(data['t1']*60))

        else:
            data = pd.read_table(filename,index_col = False,skiprows=38,delim_whitespace = True,names = [ 't1',  'T1',  'x1',  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])[['t1','T1','x1']]
            init_mass.append(data['x1'][data['T1'] >= 150].iloc[0])
            data = data[data['T1'] >= 660]
            all_data.append(data.set_index(data['t1']*60))
            #data = data[data['t1'] <= 139.5]
            #data = data[data['t1'] <= 142.5]
            data = data[::18]
            datalist.append(data.set_index(data['t1']*60))




def DERIV(t,x,k):
    answer = False
    while not answer:
        #s1 = 10**(-float(input('log_10(smoothing A): ')))
        s1=10**(-k)
        #f,a1 = plt.subplots(figsize = (18,8))
        inter = UnivariateSpline(t,x,k=3,s = s1)
        #a1.plot(t,inter.__call__(t,1)*(-1))
        #plt.show()
        #answer = int(input('1 = accept/ 0 = try again:'))
        answer=True
    return inter.__call__(t,nu=1)




datalist = []
all_data= []
read_data()

n_exp = len(datalist)

final_mass = [datalist[i]['x1'].iloc[-1] for i in range(n_exp)]
# had to multiply by c1


#Add derivative of mass w.r.t time
for n,k in enumerate([6,6,6]):
    datalist[n]['dx1_exp'] = DERIV(datalist[n].index, ((datalist[n]['x1'].iloc[0]-datalist[n]['x1'])/(datalist[n]['x1'].iloc[0]- final_mass[n])), k)
    #Change mass to convesion
    datalist[n]['x1'] = ((datalist[n]['x1'].iloc[0] -  datalist[n]['x1'])/(datalist[n]['x1'].iloc[0] - final_mass[n]))
T_in = [interp1d(datalist[n]['x1'],datalist[n]['T1']) for n in range(n_exp)]




#datalist = [datalist[i][datalist[i]['x1'] <= 0.98] for i in range(3)]  



length = [len(datalist[i]) for i in range(n_exp)]
max_val = [max(datalist[n]['dx1_exp']) for n in range(n_exp)]
print(max_val)
print(length)

"""
for n in range(n_exp):
    for i in range(length[n]):
        if datalist[n]['x1'].iloc[i] > 1.0:
            datalist[n]['x1'].iloc[i] = 1.0
"""





# FD and CD is unstable (explicit) for stiff system > need to use small timestep
# BD is more stable; implicit method of order 1
# Crank-Nicholson; implicit method of order 2
# Use sparse matrix solve scipy
# Use RK-scheme to solve equation
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
import numpy as np
from numpy import ones,diag
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize,basinhopping,newton
from numpy.linalg import norm,solve







k = 0

T_start = datalist[k]['t1'].iloc[0]
T_end = datalist[k]['t1'].iloc[-1]


#N=200
N = len(datalist[k])-1
#T_exp = np.linspace(T_start,T_end,N+1)
T_exp = datalist[k]['T1'].values + 300 # ?
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

def newton_system(func, jac, x0,p, tol = 1.e-16, max_iter=25):
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

def newton_system_trap(func, jac, x0,p, tol = 1.e-4, max_iter=20):
    x = x0
    for k in range(max_iter):
        fx = func_trap(x,p)
        if norm(fx, np.inf) < tol:          # The solution is accepted. 
            break
        Jx = jac_trap(x,p)
        delta = solve(Jx, -fx) 
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


A_ex = 10**8.4
E_ex = 145000
n_ex= 1.4

def fx(T,y):
    return A_ex*np.exp(-E_ex/(8.315*T))*(1-y)**n_ex # -0.002*(1-y)**2 #+0.001*(1-y)**2

#sol = solve_ivp(fx,(T_start,T_end),[0],dense_output=True,max_step=0.5)
#y_exp = sol.sol(T_exp)
#dx = A_ex*np.exp(-E_ex/(8.315*T_exp))*(1-sol.sol(T_exp)[0])**n_ex
dx = datalist[k]['dx1_exp'].values*60
y_exp = datalist[k]['x1'].values

b=np.zeros(N)
b[0] = 1


def of(p):
    A_m_=tridiag(-1,1,0,N,p)
    A_m_[0][0]=1
    
    Y_ = np.linalg.solve(A_m_,b)
    #print(Y_)
    #Y_nl = newton_system(func,jac,Y_,p)
    
    t0_py = time.time()
    Y_trap = newton_system_trap(func_trap,jac_trap,Y_,p)
    t1_py = time.time()
    times_py.append(t1_py-t0_py)
    
    #return np.sum((1-Y_trap-y_exp[1:])**2)
    return np.sum(((Y_trap[:-1]-Y_trap[1:])/dt-dx[2:])**2)





times_py =[]

res = minimize(of,[9,200,100],method='Nelder-mead',options={'maxiter':500})
#res=basinhopping(of, [8,150,100],niter=200,stepsize=1)




A_m=tridiag(-1,1,0,N,res.x)
A_m[0][0]=1
Y = np.linalg.solve(A_m,b)

Y_nl = newton_system(func,jac,Y,res.x)
Y_trap = newton_system_trap(func_trap, jac_trap, Y, res.x)


"""
def fx_calc(y,T):
    return 10**(res.x[0])*np.exp(-res.x[1]*1000/(8.315*T))*(1-y)**res.x[2]/100

sol_calc = odeint(fx_calc,0,T_int)

dx_calc = (10**res.x[0])*np.exp(-res.x[1]*1000/(8.315*T_int))*(1-sol_calc.T[0])**(res.x[2]/100)
"""






f,a = plt.subplots()
#a.plot(T_exp[1:],1-Y_nl,C='C0',label='calc',ls='--')
#a.plot(datalist[k]['t1'].iloc[1:],1-Y_nl,label='calc. corr.')
#a.plot(sol.t,sol.y[0],label='exp',ls=':',lw=1.5) #simulated
a.plot(datalist[k]['t1'],datalist[k]['x1'],ls=':',label='exp',lw=1.5)

a.plot(datalist[k]['t1'].iloc[1:],1-Y_trap[:],label='trap')


a2=a.twinx()

#a2.plot(T_exp[2:],N/(T_end-T_start)*(Y_nl[:-1]-Y_nl[1:])) # (f(x)-f(x-h))/h = df/dt
#a2.plot(T_exp,dx_calc)
#a2.plot(T_exp[1:],dx,ls=':',lw=1.5) # simulated
a2.plot(datalist[k]['t1'],datalist[k]['dx1_exp'],ls=':',lw=1.5)

a2.plot(datalist[k]['t1'].iloc[2:],1/60*N/(T_end-T_start)*(Y_trap[:-1]-Y_trap[1:]))
#a2.plot(datalist[k]['t1'].iloc[2:],1/60*(Y[:-1]-Y[1:])/(T_end-T_start)*N,lw=0.8)
a.legend()


