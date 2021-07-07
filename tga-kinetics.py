import sys
import os
import tkinter as tk
from tkinter import Tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilenames
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk)
from scipy.interpolate import UnivariateSpline
import pylightxl as xl
from numpy import ones,diag
from scipy.optimize import minimize,basinhopping,newton
from numpy.linalg import norm,solve
import time
from scipy.sparse import diags,csc_matrix
from scipy.sparse.linalg import spsolve
import time
import datetime
import cma

def nonlinear(x,n): # not used in program.
    # if 40 < n < 92, empirical function to get better starting position
    # for Y_linear to give to newton_system_trap. Not well done,is slower and
    # yields worse results. Should be re-parameterized with another function
    return np.exp(7.457309871653051 -0.09198587680969814*n)*x**(3.239250503132271 -0.02799505738572956*n)*np.exp(-(12.017401311759613 -0.07096770044261062*n)*x)


infile_columns = ['t','T','m_exp']
data = {'t':[],'T':[],'m_exp':[],'x_exp':[]}
data_calc = {'t':[],'T':[],'x_exp':[],'dx_exp':[]}

info =  {'n_data':0,'filenames':[],'current_exp':0,
         'T_start':[],'T_end':[],'t_start':[],'t_end':[],
         's.factor':[],'n_points':[],'mass norm. temp.':[],
         'max':[]}


root = Tk()
root.geometry('850x475')
root.title('tga-kinetics.py')

tab_control = ttk.Notebook(root,)


tabs = [tk.Frame(tab_control) for i in range(5)]
tab_labels = ['Input','Kinetic options','Optimization options','Results','Export...']
for tab,label in zip(tabs,tab_labels):
    tab_control.add(tab,text=label)
tab_control.pack(side=tk.TOP,anchor='nw',ipady=0,ipadx=0)




        
################
"""Input tab"""
################        
def read_ascii(filename):
    # Read data from regular ascii format documents
    loaded_data = np.loadtxt(filename)
    for column,key in enumerate(infile_columns):
        data[key].append(loaded_data.T[column])

def read_excel(filename):
    # Read data from excel documents
    db = xl.readxl(filename)
    loaded_data = [] # tmp storage t,T,x
    for i,row in enumerate(db.ws(ws='Sheet1').rows):
            loaded_data.append(row)
    loaded_data = np.array(loaded_data[1:])
    for column,key in enumerate(infile_columns):
        data[key].append(loaded_data.T[column])
        
def normalize_mass(i):
    # calculate normalized mass reaction rate or conversion
    #e = np.where(data['T'][i]<float(info['T_end'][i]))[0][-1] # end index
    s = np.where(data['T'][i]>float(info['mass norm. temp.'][i]+273.15))[0][0] # start index
    # add possibility to choose where to calc conversion
    # add possibility to calculate normalized mass instead of conv.
    
    data['x_exp'][i] = (data['m_exp'][i])/(data['m_exp'][i][s])
   

def ask_to_close_window():
    # Popoup to ask closing the window if data allready exists. 
    error_window = Tk()
    error_window.geometry('150x150')
    error_text = " \nClose and restart the program to load new experimental data\n"
    popup = tk.Message(error_window,text=error_text)
    popup.grid(row=1,column=1)

def Try_read_data():
    if info['n_data']: # if data llready exists, ask to restart the program
        ask_to_close_window()
    else:
        Read_data()
        t0['select_data_btn'].grid_remove()

    

def Read_data():
    filenames = []
    """
    Read data from a text file and appends to data dictionary. 
    Data structure should be:
        # t   T (C)      x      dx 
         t_1   T_1    x_1    dx_1
         t_2   T_2    x_2    dx_2
         t_3   T_3    x_3    dx_3
              ...
         t_n   T_n    x_n    dx_n

    """
    files = askopenfilenames()
    clear_info = [0,[],0,[],[],[],[],[],[],[],[],[],[],[]]
    #if info['n_data']: # if data llready exists, ask to restart the program

    for filename in files:
        filenames.append(filename.split('/')[-1].split('.')[0]) # for filename display
        if 'xls' in filename.split('.')[-1]:
            read_excel(filename)
        else:
            read_ascii(filename)
    
    extract_info_from_data(filenames)
    Plot_raw_data()
    write_info_text()
    choose_experiment()


        
def extract_info_from_data(filenames):
    info['n_data'] = len(data['T'])
    info['filenames']=filenames
    #add derivative
    data['dx_exp'] = {}
    for n in range(info['n_data']):
        data['T'][n] += 273.15
        data['x_exp'].append(1)
        info['s.factor'].append(1e-4)
        info['T_start'].append(data['T'][n][0])
        info['t_start'].append(data['t'][n][0])
        info['T_end'].append(data['T'][n][-1])
        info['t_end'].append(data['t'][n][-1])
        info['mass norm. temp.'].append(100)
        normalize_mass(n)
        spl =  UnivariateSpline(data['t'][n],data['x_exp'][n],s=info['s.factor'][n])
        data['dx_exp'][n] = spl.__call__(data['t'][n],nu=1)*(-1) # first derivative
        info['max'].append(max(data['dx_exp'][n]))
        #info['n_points'].append(len(data['t'][n]))
        info['n_points'].append(100)


def cut(n):
    # Determine the shortest array from t_/T_start and t_/T_end and cut data 
    #accordingly. Returns the cut indices start_indice, end_indice

    i_min = 0
    i_max = len(data['t'][n])
    for var in ['t','T']:
        tmp_min = np.where(data[var][n]>=info[var+'_start'][n])[0]
        tmp_max = np.where(data[var][n]>=info[var + '_end'][n])[0]
        
        if len(tmp_min):
            i_min = max(i_min,tmp_min[0])
        if len(tmp_max):
            i_max = min(i_max,tmp_max[0])
    return i_min,i_max
    
def Plot_raw_data():
    """
    Plot the raw data in the data dict.
    """
    
    i = info['current_exp']
    f,ax = plt.subplots(1,figsize=(6,3))
    s,e = cut(i) # s = start, e = end values to plot
    
    ax.plot(data['t'][i][s:e],data['dx_exp'][i][s:e],label='dm_norm/dt',c='k',ls='--')
    ax.set(xlabel='time',ylabel='reaction rate')
    aT = ax.twinx()
    aT.plot(data['t'][i][s:e],data['T'][i][s:e],label='T (K)',c='C3',ls='--')
    aT.set(ylabel='temperature (K)')
    f.legend(loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(0.5,1.025))
    plt.tight_layout()
    #plt.show()
    # Embed plot in tkinter
    if t0['fig_exp']:
        t0['fig_exp'].get_tk_widget().update()
    t0['fig_exp'] = FigureCanvasTkAgg(f,master = tabs[0])   
    t0['fig_exp'].draw() 
    t0['fig_exp'].get_tk_widget().grid(row=1,rowspan=16,column=2,ipady=20,ipadx=4,padx=4,sticky=tk.W)
    t0['plot_header'] = tk.Label(tabs[0],text='-- '+ info['filenames'][info['current_exp']] +' --')
    t0['plot_header'].grid(row=0,column=2)
    # add toolbake
    toolbarFrame = tk.Frame(tabs[0])
    toolbarFrame.grid(row=18,column=2,sticky=tk.W)

    toolbar = NavigationToolbar2Tk(t0['fig_exp'], toolbarFrame)
    toolbar.pack()
    toolbar.update()
    plt.close() # do not display figure in spyder when using interactive mode

def Accept_data():
    # Read in data from entry for info tables
    n = info['current_exp']

    for row,var in enumerate(info_vars):            
        val = float(t0['etr'][var].get())
        info[var][n] = val #accept all values as given in entry
        
    s,e = cut(n)
    for var in info_vars:
            if '_start' in var:    
                val = max(info[var][n],data[var[0]][n][0],data[var[0]][n][s])
                info[var][n]=val #update with correct minimum limit
                #val = data[var[0]][n][s]
            elif '_end' in var:
                val = val = min(info[var][n],data[var[0]][n][-1],data[var[0]][n][e])
                info[var][n]=val #update with correct maximum limit
            elif 'n_points' in var:
                val = int(len(data['t'][n])/(max(int(len(data['t'][n])/info[var][n]),1)))-1
                info[var][n]=val
    normalize_mass(n)
    spl =  UnivariateSpline(data['t'][n],data['x_exp'][n],s=info['s.factor'][n])
    data['dx_exp'][n] = spl.__call__(data['t'][n],nu=1)*(-1) # first derivative
    Plot_raw_data()
    create_equally_spaced_data()
    write_info_text()
    
def create_equally_spaced_data():
    # Create equally spaced datapoints according to input from 
    # info['n_points'] and save this to data_calc, which will be used 
    # to perform the actual calculations because implicit integration
    # can be expensive. If over 300 points sparse matrix can be used. 
    if not data_calc['t']: # if no exisiting data, create new dict
        for var in ['t','T','m_exp','dx_exp']:
            data_calc[var] = [[]]*info['n_data']
    
    n_d = info['current_exp']
    s,e = cut(n_d)
    length = len(data['t'][n_d][s:e])
    step = max(int(length/info['n_points'][n_d]),1)
    
    for var in ['t','T','m_exp','dx_exp']:
        data_calc[var][n_d]=data[var][n_d][s:e:step]
    info['max'][n_d] = max(data_calc['dx_exp'][n_d])
    info['n_points'][n_d] = len(data_calc['t'][n_d])
        

def write_info_text():
    # Write information extracted from experimental data for display and
    # to edit the input data
    n = info['current_exp']
    s,e = cut(n)
    for row,var in enumerate(info_vars):
        txt = var
        t0['txt'][var] = tk.Text(tabs[0],width='19',height='1')
        if var =='s.factor':
            txt ='smoothing, (1e-2)'
        elif var == 'mass norm. temp.':
            txt += (',K')
        t0['txt'][var].insert('1.0',txt)
        t0['txt'][var].grid(row=1+row,column=0)
        
        t0['etr'][var] = tk.Entry(tabs[0],width='8')
        t0['etr'][var].grid(row=1+row,column=1)
        # Needed to avoid getting many deciamls in 4some entry widgets
        if var == 'n_points' or var =='s.factor':
            val = str(info[var][n])
        else:
            val = str(round(info[var][n],3))
            
        t0['etr'][var].insert(0,val)

    
def choose_experiment():
    t0['cb'] = ttk.Combobox(tabs[0],textvar=t0['cb_var'],width=5,state='readonly')
    t0['cb']['values'] = tuple([str(i+1) for i in range(info['n_data'])] )
    t0['cb'].current(0)
    t0['cb'].grid(row=len(info_vars)+2,column=1)
    
    

def Change_experiment():
    info['current_exp'] = t0['cb'].current()
    write_info_text()
    Plot_raw_data()
    

info_vars = ['T_start','T_end','t_start','t_end','s.factor','n_points','mass norm. temp.']
#tabs[0] dict of all widgets
t0 = {'text_exp':{},'fig_exp':{},'txt':{},'etr':{},'cb':0,'cb_var':tk.StringVar()}
t0['select_data_btn'] = tk.Button(tabs[0],text=' '*6+'Select data'+' '*6,command=Try_read_data)
t0['select_data_btn'].grid(row=0,column=0,sticky=tk.W)

tk.Button(tabs[0],text='Change experiment',command=Change_experiment).grid(row=len(info_vars)+2,column=0)
tk.Button(tabs[0],text='{:>12s}'.format(' '*5+ 'Accept data' + ' '*6),command=Accept_data,fg='green').grid(row=len(info_vars)+1,column=0,sticky=tk.W)

##########################
"""Kinetic options tab"""
##########################
def Accept_kinetics():
    hide_show_kinetic_options()
    
def get_kinetic_options():
    t1['n_part']  = tk.IntVar()
    
    tk.Label(tabs[1],text='{:<29}'.format('Number of partial components:')).grid(row=0,column=0,sticky=tk.W)
    #tk.Label(tabs[1],text='{:<29}'.format('Accept n_partial components.:')).grid(row=1,column=0,columnspan=3,sticky=tk.E)
    for n in range(3):
        tk.Radiobutton(tabs[1],text=str(n+1),variable=t1['n_part'],value=n,command=Accept_kinetics).grid(row=0,column=n+1)
    #for n,var in enumerate(['A','E','n']):
    #    t1['common_params'][var] = tk.IntVar()
    #    btn = tk.Checkbutton(tabs[1],text=var,variable=t1['common_params'][var],onvalue=1,offvalue=0,state=tk.DISABLED)
    #    btn.grid(row=1,column=n+1)
    #    btn.select()
    #tk.Button(tabs[1],text='\u221a',command=Accept_kinetics,fg='green').grid(row=1,column=4)

    
def write_kinetic_options():
    description={'A':['Frequency factor','log10'],'E':['Activation energy','kj/mol'],
                'n':['Reaction order','-'],'c':['Scaling factor','-']}
    init_vals = [8,200,1,0.2] # A, E , n
    steps =[0.25,20,0,0]
    #defaults = {var:[val]*n_exp for var,val in zip(description.keys(),init_vals)}
    #defaults = {var:[val] for var,val in zip(description.keys(),init_vals)}
    defaults_part={var:[init_vals[i]+k*steps[i] for k in range(3)] for i,var in enumerate(description.keys())}
    mult = len(init_vals) # row multiplier for construction of table
    # Partial component iterator to construct table
    for n in range(3):
        t1['part_txt'][n] = tk.Label(tabs[1],text='Initial values partial component nr. '+str(n+1)+ ' | Common experimental parameter:')
        t1['part_txt'][n].grid(row=2+(mult+1)*n,column=0,columnspan=5)

    # Experiment iterator to construct table
    for n in range(3):
        for row,var in enumerate(description.keys()):
            t1['description'][var][n]  = tk.Label(tabs[1],text='{:<22} {:>5}'.format(description[var][0],description[var][1]))
            t1['description'][var][n].grid(sticky=tk.W,row=row+3+(mult+1)*n,column=0)
            t1[var][n]=tk.Entry(tabs[1],width='5')
            t1[var][n].grid(row=row+3+(mult+1)*n,column=1)
            t1[var][n].insert(0,defaults_part[var][n])
            t1['fixed'][var][n] = tk.IntVar()   
            t1['fixed_btns'][var][n] = tk.Checkbutton(tabs[1],text='constant',variable=t1['fixed'][var][n],onvalue=1,offvalue=0)
            t1['fixed_btns'][var][n].grid(row=row+3+(mult+1)*n,column=2)
            
            t1['common_params'][var][n] = tk.IntVar()
            t1['common_btns'][var][n] = tk.Checkbutton(tabs[1],text=var,variable=t1['common_params'][var][n],onvalue=1,offvalue=0)
            t1['common_btns'][var][n].grid(row=2+(mult+1)*n,column=6+row)
            t1['common_btns'][var][n].select()

def hide_show_kinetic_options():
    # Hide or show kinetic options for partial components
    # when checkbutton is clicked. Hide all then show n_part of widgets
    for n in range(3):
        t1['part_txt'][n].grid_remove()
        for var in ['A','E','n','c']:
            t1[var][n].grid_remove()
            t1['description'][var][n].grid_remove()
            t1['fixed_btns'][var][n].grid_remove()
            t1['common_btns'][var][n].grid_remove()
    for n in range(t1['n_part'].get()+1):
        t1['part_txt'][n].grid()
        for var in ['A','E','n','c']:
            t1[var][n].grid()
            t1['description'][var][n].grid()
            t1['fixed_btns'][var][n].grid()
            t1['common_btns'][var][n].grid()
t1 = {'A':{},'E':{},'n':{},'c':{},'common_params':{'A':{},'E':{},'n':{},'c':{}},'n_part':1,'part_txt':{},
      'description':{'E':{},'A':{},'n':{},'c':{}},'fixed':{'A':{},'E':{},'n':{},'c':{}},
      'fixed_btns':{'A':{},'E':{},'n':{},'c':{}},'common_btns':{'A':{},'E':{},'n':{},'c':{}}}
get_kinetic_options()
write_kinetic_options()
hide_show_kinetic_options()

###############################
"""Optimization options tab"""
###############################
def write_optimization_options():
    defaults = dict(t2)
    description={
                 'method':'Optimiztion method',
                 'maxiter':'Maximum optimization iterations\nCMA-ES: function tolerance (e.g. 1e-7)',
                 'newton_tol':'Non-linear eq. solution tolerance',
                 'newton_maxiter':'Maximum non-linear eq. solving iterations'}
    for n,key in enumerate(description.keys()):
        tk.Label(tabs[2],text=description[key]).grid(row=n,column=0,sticky=tk.W)
        if key =='method':
            t2[key] = ttk.Combobox(tabs[2],textvar=t2['method_var'],width=12,state='readonly')
            t2[key]['values'] = tuple(method for method in ['Nelder-mead','CMA-ES','Powell'] )
            t2[key].current(0)
        else:
            t2[key] = tk.Entry(tabs[2],width=10)
        t2[key].grid(row=n,column=1)
        t2[key].insert(0,defaults[key])
t2 = {'method_var':tk.StringVar(),'method':'Nelder-mead','newton_tol':1e-3,
      'newton_maxiter':5,'maxiter':500}

def run_optimization():
    print('Calculating...')
    t2['N'] = [len(data_calc['t'][n_d])-1 for n_d in range(info['n_data'])]
    t2['n_p']=t1['n_part'].get()+1
    t2['n_d']=info['n_data']
    t2['dt'] = [(info['t_end'][i] - info['t_start'][i])/t2['N'][i] for i in range(info['n_data'])]
    p = calc_p_num()
    #for n_p in range(t2['n_p']):   # add scaling factor x_t = p[-1]*x_1 + p[-2]*x_2
    #    p.append(1/t2['n_p'])      # could set it so that x1 = p[-1],x2=1-p[-1] and they sum to one
    
    t3['nonlinear'] = 0 #nonlinear-equation solving time
    
    if t2['method_var'].get()=='CMA-ES':
        xopt,res = cma.fmin2(of, p, 0.5,options={'tolfun':t2['maxiter'].get()})
        class result:
            x = xopt
            fun = res.result[1]
        t2['res'] = result
  
    else:
        t2['res'] = minimize(of,p,method=t2['method_var'].get(),options={'maxiter':int(t2['maxiter'].get())})

    calc_from_res(t2['res'].x)
    t2['p_letters'] = calc_p()
    write_result(t2['p_letters'],t2['res'])


            
def of(p):
    # The objective function to be minimized.
    r = calc_r_num(p)
    y=0
    for i in range(t2['n_d']):
        y_p = 0
        for n in range(t2['n_p']):
            A = tridiag(t2['N'][i],[r[0][n][i],r[1][n][i],r[2][n][i]*100],t2['dt'][i],data_calc['T'][i])
            A[0][0] = 1
            b=np.zeros(t2['N'][i])
            b[0] = 1 # initial condition: y0 = 1    
            Y_ = solve(A,b)
            t0_py = time.time()
            Y_trap = newton_system_trap(func_trap, jac_trap, Y_, [r[0][n][i],r[1][n][i],r[2][n][i]*100],
                                t2['dt'][i],data_calc['T'][i],t2['N'][i],tol=float(t2['newton_tol'].get()),max_iter=int(t2['newton_maxiter'].get()))
            t1_py = time.time()
            t3['nonlinear'] += (t1_py-t0_py)
            
            #y +=np.sum((data_calc['x_exp'][i][:-1]-Y_trap)**2)
            
            #y_p+=(r[3][n][i]*10**(r[0][n][i])*np.exp(-r[1][n][i]*1000/(8.3145*data_calc['T'][i][:-1]))*Y_trap**(r[2][n][i]))
            # The above fluctuates at high conversion values
            
            y_p+=(r[3][n][i]*(Y_trap[:-1]-Y_trap[1:])/t2['dt'][i])
            
        y += np.sum(((y_p-data_calc['dx_exp'][i][:-2])**2))/(t2['N'][i]*info['max'][i]**2)
    return y


def calc_p_num():
    # Calculate the initial value list p to pass to scipy.minimize function. 
    # One has to take into account fixed values when creating the list
    p=[]
    for n_v,var in enumerate(['A','E','n','c']):
        for n_p in range(t2['n_p']):
            if t1['common_params'][var][n_p].get() and not t1['fixed'][var][n_p].get():
                    p.append(float(t1[var][n_p].get()))
            elif not t1['fixed'][var][n_p].get():
                    for n_d in range(t2['n_d']):
                        p.append(float(t1[var][n_p].get()))
    return p

def calc_p():
    p=[]
    for n_v,var in enumerate(['A','E','n','c']):
        for n_p in range(t2['n_p']):
            if t1['common_params'][var][n_p].get() and not t1['fixed'][var][n_p].get():
                    p.append(var+str(n_p))
            elif not t1['fixed'][var][n_p].get():
                    for n_d in range(t2['n_d']):
                        p.append(var+str(n_p)+str(n_d))
    return p
    
    

def calc_r_num(p):
    # calculate a 3D list such that r[var][n_p][n_d],where var is 'A':0,'E':1,'n':0, 
    # n_p is partial component, and n_d is experiment a.k.a data number. The 
    # r 3D list then yields the correct value from either the constant value
    #for this variable or the value p[i] from the scipy.minimize list p. Negative
    # values not allowed in optimization. Thus abs() is added. 
    
    i = 0 # keep track of index i
    r = []
    for n_v,var in enumerate(['A','E','n','c']):
        m = []
        for n_p in range(t2['n_p']):
            n = []
            if t1['common_params'][var][n_p].get():
                if t1['fixed'][var][n_p].get():
                    for n_d in range(t2['n_d']):
                        n.append(float(t1[var][n_p].get()))
                else:
                    for n_d in range(t2['n_d']):
                        n.append(abs(p[i]))
                    i+=1
                m.append(n)
            else:
                if t1['fixed'][var][n_p].get():
                    for n_d in range(t2['n_d']):
                        n.append(float(t1[var][n_p].get()))
                else:
                    for n_d in range(t2['n_d']):
                        n.append(abs(p[i]))
                        i+=1
                m.append(n)
        r.append(m)
    return r


def calc_from_res(p):
    # calculate conversion, error% etc from optimization results
    # and store values to dict
    r = calc_r_num(p)
    print('\nResult:\n',p)
    t3['err%']=[]
    for i in range(t2['n_d']):
        Y_part = 0
        DY_part = 0
        t3['dx_calc'][i] = []
        t3['x_calc'][i] = []
        for n in range(t2['n_p']):
            s,e = cut(i)
            A = tridiag(t2['N'][i],[r[0][n][i],r[1][n][i],r[2][n][i]*100],t2['dt'][i],data_calc['T'][i])
            A[0][0] = 1
            b=np.zeros(t2['N'][i])
            b[0] = 1 # initial condition: y0 = 1    
            Y_ = solve(A,b)
            Y_trap = newton_system_trap(func_trap, jac_trap, Y_, [r[0][n][i],r[1][n][i],r[2][n][i]*100],
                                t2['dt'][i],data_calc['T'][i],t2['N'][i])
            
            t3['dx_calc'][i].append(r[3][n][i]*(Y_trap[:-1]-Y_trap[1:])/t2['dt'][i])
            t3['x_calc'][i].append(r[3][n][i]*Y_trap)
            
            Y_part += r[3][n][i]*Y_trap
            DY_part +=t3['dx_calc'][i][n]
        t3['err%'].append(np.sqrt(np.sum((DY_part-data_calc['dx_exp'][i][:-2])**2)/(t2['N'][i]*info['max'][i]**2))*100)

    plot_results(t2['res'].x)
        
    print('Nonlinear solution time:',round(t3['nonlinear'],3),'s.')


write_optimization_options()
run_btn = tk.Button(tabs[2],text='Run calculation',command=run_optimization,fg='green')
run_btn.grid(row=10,column=0,sticky=tk.W)

###############
#MATRIX METHODS
################
def tridiag(N,p,dt,T):
    #T_s/T_e T_start/T_end, T = experimental temp.
    # p = list of E A n in order [E,A,n]
    e = ones(N)       # array [1,1,...,1] of length N
    A = (-1)*diag(e[1:],-1)+diag(1+dt*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T[1:])))
    return A

def jac_trap(y,p,dt,T,N):
    y = y.clip(1e-16) # nexxessary
    e = ones(N)       # array [1,1,...,1] of length N
    A = diag(-1+e[1:]*0.5*dt*(10**(p[0]))*(p[2]/100)*np.exp(-(p[1]*1000)/(8.315*T[:-2]))*y[:-1]**(p[2]/100-1),-1)+diag(1+0.5*dt*(10**(p[0]))*(p[2]/100)*np.exp(-(p[1]*1000)/(8.315*T[1:]))*y**(p[2]/100-1))
    A[0][0]=1
    return A

def newton_system_trap(func, jac, x0,p,dt,T,N, tol =1e-4, max_iter=20):
    x = x0
    for k in range(max_iter):
        fx = func(x,p,dt,T)
        if norm(fx, np.inf) < tol:          # The solution is accepted. 
            break
        Jx = jac(x,p,dt,T,N)
        delta = solve(Jx, -fx) 
        x = x + delta 
        x = x.clip(0) #if x<0 replace by 0 to avoid numerical issues
    return x

def func_trap(x,p,dt,T):
    #x = x.clip(1e-12) # add if errors 
    y = np.array([-x[:-1]+x[1:]+0.5*dt*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T[1:-1]))*(x[1:])**(p[2]/100)+ 0.5*dt*10**(p[0])*np.exp(-(p[1]*1000)/(8.315*T[:-2]))*(x[:-1])**(p[2]/100)])
    y = np.insert(y,0,0) #y_0 - 1 = 1 - 1 = 0
    return y


#################
"""Results tab"""
#################

def tabnew(j):
    # write either a tab or newline character in result text 
    # should be placed in t3['results'] and j updated with += 1 directly after
    if j%2 == 0:
        return ' '*4
    elif j%2 == 1:
        return '\n'
    
def write_result(p,res):
    print('OF = {:.3e}'.format(res.fun))
    t3['results'].delete('1.0','end')
    #Estimated value results
    t3['results'].insert('end','   Estimated:\n')
    j = 0
    for p_i,res_i in zip(p,res.x):
        t3['results'].insert('end','{:<3s}={:>7.3f}'.format(p_i,abs(res_i))+tabnew(j))
        j+=1
    #Fixed values
    if tabnew(j)=='\n':
        next_del = '\n\n   '
    else:
        next_del = '\n   '
    t3['results'].insert('end',next_del+'Fixed:\n')
    if sum([int(t1['fixed'][var][n_p].get()) for var in ['A','E','n','c'] for n_p in range(t2['n_p']) ]) == 0:
        t3['results'].insert('end','None\n')
    else:
        for var in ['A','E','n','c']:
            for n_p in range(t2['n_p']):
                    if t1['fixed'][var][n_p].get():
                        t3['results'].insert('end','{:<3s}={:>7.3f}'.format(var+str(n_p),float(t1[var][n_p].get()))+tabnew(j))
                        j+=1
    #Function result
    t3['results'].insert('end','\nOF={:>7.2e}\n'.format(res.fun))
    # error in %
    #t3['results'].insert('end','\nErr%={:>7.2f}'.format(np.sqrt(res.fun)*100/info['n_data']))
    
def change_result_plot_experiment():
    t3['fig_result'][t3['current_res_exp']].get_tk_widget().grid_remove()
    t3['plot_header'][t3['current_res_exp']].grid_remove()
    t3['current_res_exp'] +=1
    t3['current_res_exp'] %=t2['n_d']
    t3['fig_result'][t3['current_res_exp']].get_tk_widget().grid()
    t3['plot_header'][t3['current_res_exp']].grid()

    
    
    
def plot_results(p):
    #p = t2['res'].x
    c_np = ['C2','C4','C6'] # partial component colors
    t3['fig_result'] = {}
    t3['plot_header']={}
    for i in range(t2['n_d']):
        f_r,(ax_r) = plt.subplots(1,1,figsize=(6,3))
        s,e = cut(i) # s = start, e = end values to plot
        ax_r.plot(data['t'][i][s:e],data['dx_exp'][i][s:e],label='exp',c='k',ls='--')
        DY = 0
        for n_p in range(t2['n_p']): # partial component conversion data
            DY += t3['dx_calc'][i][n_p]
            ax_r.plot(data_calc['t'][i][:-2],t3['dx_calc'][i][n_p],label='calc'+str(n_p),c=c_np[n_p],ls='--')
        
        if t2['n_p']> 1: # add sum of partial components to plot only if > 1
            ax_r.plot(data_calc['t'][i][:-2],DY,label='sum',c='C3')
        ax_r.set(xlabel='time',ylabel='reaction rate')
        f_r.legend(loc='upper center',frameon=False,ncol=t2['n_p']+2,bbox_to_anchor=(0.5,1.025))
        plt.tight_layout()
        # add figure to tkinter
        
        t3['fig_result'][i] = FigureCanvasTkAgg(f_r,master = tabs[3])   
        t3['fig_result'][i].get_tk_widget().grid(row=1,columnspan=2,column=1,ipady=20,ipadx=4,padx=4,sticky=tk.W)
        t3['fig_result'][i].draw()
        t3['plot_header'][i] = tk.Label(tabs[3],text='-- '+ info['filenames'][i] +' --\n'+
                                     'err%={:3.2f}'.format(t3['err%'][i]))
        t3['plot_header'][i].grid(row=0,column=1)
        plt.close()
        t3['fig_result'][i].get_tk_widget().grid_remove()
        t3['plot_header'][i].grid_remove()
        toolbarFrame = tk.Frame(tabs[3])
        toolbarFrame.grid(row=18,column=1,sticky=tk.W,columnspan=2)
        toolbar = NavigationToolbar2Tk(t3['fig_result'][i], toolbarFrame)
        toolbar.pack()
        toolbar.update()
    t3['fig_result'][t3['current_res_exp']].get_tk_widget().grid()
    t3['plot_header'][t3['current_res_exp']].grid()
    

t3 = {'results':tk.Text(tabs[3],height=27,width=29),'dx_calc':{},'x_calc':{},
      'current_res_exp':0}

t3['results'].grid(row=0,column=0,rowspan=4)
t3['results'].configure(font=("Times new roman", 10))
tk.Button(tabs[3],text='Next exp.',command=change_result_plot_experiment).grid(row=0,column=2,sticky=tk.W)

###################
"""--Export tab-"""
###################
def export_optimization_results(p,res):
    # fuction to write estimated parameters, fixed parameters and errors in %.
    s = '    OPTIMIZATION RESULTS'
    #Estimated value results
    s+=   '\n\nEstimated:\n'
    j = 0
    for p_i,res_i in zip(p,res.x):
        s+='{:<3s}={:>7.3f}'.format(p_i,abs(res_i))+tabnew(j)
        j+=1
    #Fixed values'
    s+='\n\nFixed:\n'
    if sum([int(t1['fixed'][var][n_p].get()) for var in ['A','E','n','c'] for n_p in range(t2['n_p']) ]) == 0:
        s+='None\n'
    else:
        j=0
        for var in ['A','E','n','c']:
            for n_p in range(t2['n_p']):
                    if t1['fixed'][var][n_p].get():
                        s+='{:<3s}={:>7.3f}'.format(var+str(n_p),float(t1[var][n_p].get()))+tabnew(j)
                        j+=1
    #Error percent and objective function result
    s+='\n\nerror%\n'
    for i in range(t2['n_d']):
        s+='exp. '+str(i+1)+': '+str(round(t3['err%'][i],2))+'%\n'
    s+='\nOF={:>7.2e}\n\n'.format(res.fun)
    return s

def delim(i):
    if i%3 == 0:
        return '\n'
    else:
        return '\t'
    
def export_experimental_info():
    # write the experimental information used to calculate the results
    s ='    EXPERIMENTAL INFORMATION'
    for n_d in range(t2['n_d']):
        s+='\n\nExp. ' + str(n_d+1)+ ' : '+info['filenames'][n_d]+'\n\n'
        for key in info_vars+['max']:
            if key == 'max':
                s+= '{:<16} : {:<12.3e}'.format(key+ ' react. rate',info[key][n_d])+'\n'
            else:
                 s+= '{:<16} : {:<12}'.format(key,str(info[key][n_d]))+'\n'
    return s
            
            
            
    
def save_info():
    # save data info
    infofile = open(t4['export_entry'].get()+'_info.txt','w')
    infofile.write(export_optimization_results(t2['p_letters'], t2['res']))
    infofile.write(export_experimental_info())
    infofile.close()
    
def create_header(header_file_list):
    header = ''
    for header_file in header_file_list:
        for header_i in header_file:
            header+=header_i + ' '
    return header

def save_data(filename): 
    # save experimental and calculated data
    fmt = ['%.3f','%10.3f','%10.3e']+['%10.3f']*t2['n_p']+['%10.3e']*t2['n_p']
    
    for n_d in range(t2['n_d']):
        header = ' time'+'  m_norm_exp'+str(n_d)+'  dm_exp'+str(n_d) + ''
        for n_p in range(t2['n_p']):
            header +='   m_calc'+str(n_d)+str(n_p)
            
        for n_p in range(t2['n_p']):
            header +='  dm_calc'+str(n_d)+str(n_p)
            
        dx_calc = np.c_[[t3['dx_calc'][n_d][n_p] for n_p in range(t2['n_p'])]].T
        x_calc = np.c_[[t3['x_calc'][n_d][n_p] for n_p in range(t2['n_p'])]].T
        to_save = np.c_[data_calc['t'][n_d][:-2],data_calc['m_exp'][n_d][:-2].T,data_calc['dx_exp'][n_d][:-2].T,
                         x_calc[:-1],dx_calc]
                        


        np.savetxt(filename+'_data_'+str(n_d)+'.txt',to_save,fmt=fmt,header=header)
    #np.savetxt(t4['export_entry'].get()+'_data.txt',to_save,header=header)

    
def save_text():
    save_info()
    save_data(t4['export_entry'].get())
    print('Saved to '+t4['export_entry'].get() + '_info.txt and\n'+t4['export_entry'].get() +' _(d)x_data.txt')



t4 = {'export_entry':tk.Entry(tabs[4],text='filename',width=20)}
t4['export_entry'].grid(row=0,column=1)
t4['export_entry'].insert('1',str(datetime.datetime.now()).split('.')[0].split(' ')[0])
tk.Button(tabs[4],text='Save data',command=save_text).grid(row=1,column=0)
tk.Label(tabs[4],text='filename:').grid(row=0,column=0)




##############
"""--END--"""
##############
root.mainloop()
#input('Press <Enter> to end the program\n') # avoid closing after loading data in windows







