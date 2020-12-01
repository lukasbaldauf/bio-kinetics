import tkinter as tk
from tkinter import Tk
from tkinter import ttk
from tkinter.filedialog import askopenfilenames
import numpy as np
import matplotlib.pyplot as plt

    
    

data = {'t':[],'T':[],'x_exp':[],'dx_exp':[]}
info =  {'n_data':0}
root = Tk()
root.geometry('900x700')
root.title('Bio-kinetics.py')

tab_control = ttk.Notebook(root)


tabs = [tk.Frame(tab_control) for i in range(5)]
tab_labels = ['Input','Kinetic options','Optimization options','Results','Export...']
for tab,label in zip(tabs,tab_labels):
    tab_control.add(tab,text=label)
tab_control.pack(expand=1, fill="both")




        
################
"""Input tab"""
################
def Read_data():
    """
    Read data from a text file and appends to data dictionary. 
    Data structure should be:
        # t     T      x      dx 
         t_1   T_1    x_1    dx_1
         t_2   T_2    x_2    dx_2
         t_3   T_3    x_3    dx_3
              ...
         t_n   T_n    x_n    dx_n

    """
    filenames = askopenfilenames()
    if info['n_data']:
        for key in data.keys():
            data[key] = []
    for filename in filenames:
        loaded_data = np.loadtxt(filename)
        for column,key in enumerate(data.keys()):
            data[key].append(loaded_data.T[column])
    info['n_data'] = len(data['T'])

def Plot_raw_data():
    """
    Plot the raw data in the data dict.
    """
    f,a = plt.subplots(1,info['n_data'],figsize=(info['n_data']*3,3))
    if info['n_data']==1:
        axs = [a]
    else:
        axs = a.flatten()
    for i,ax in enumerate(axs):
        ax.plot(data['t'][i],data['x_exp'][i],label='x')
        ax.set(xlabel='time',ylabel='conversion')
        aT = ax.twinx()
        aT.plot(data['t'][i],data['T'][i],c='C3',label='T')
        aT.set(ylabel='temperature')
        if i ==0:
            f.legend(loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(0.5,1.035))
    plt.tight_layout()
    plt.show()

tk.Label(tabs[0],text='{:>16s}'.format('Select input data')).grid(row=0,column=0)
tk.Button(tabs[0],text='{:>12s}'.format('Select data'),command=Read_data).grid(row=0,column=1)

tk.Label(tabs[0],text='{:>16s}'.format('Plot raw data')).grid(row=1,column=0)
tk.Button(tabs[0],text='{:>12s}'.format('Plot raw data'),command=Plot_raw_data).grid(row=1,column=1)

    
root.mainloop()





