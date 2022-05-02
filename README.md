# tga-kinetics

## Use tga-kinetics.jl instead for better performance and more robust numerical methods: https://github.com/lukasbaldauf/tga-kinetics.jl

**Tga-kinetics** is a simple python script that estimates kinetic parameters from thermogravimetric time, temperature and sample mass data. It aims to simplify the kinetic analysis of TGA data for its users through a graphical user interface. It was initially intended for biochar gasification but is also applicable to other TGA experiments. A set of simulated test-data experiments are supplied to test the program and illustrate the input-data structure. Excel files are also supported as input, given that the data structure matches that of the test-data with constant time steps. 


![Screenshot](https://github.com/lukasbaldauf/tga-kinetics/blob/main/program_screenshot.png)

The parameter estimation is based on the Arrhenius type rate equation:

    -dm/dt*(1/c) = dx/dt = A*exp(-E/(R * T))*x^n                 (1)

where *dm/dt* is the derivative of the normalized mass of the sample with respect to time, *c* is a scaling factor to convert the conversion rate *dx/dt* to  *-dm/dt*, *A* is the frequency-factor, *E* is the activation energy, *R* the gas constant, *x* is the conversion (i.e. the  fraction of sample left to react which varies from 1 to 0) and *n* is the reaction order (which does not need to be an integer). For additional information, see e.g.: 

- Wang, L., Li, T., Várhegyi, G., Skreiberg, Ø., & Løvås, T. (2018).  
*CO2 gasification of chars prepared by fast and slow pyrolysis from wood and forest residue: a kinetic study.*  
Energy & Fuels, 32(1), 588-597. 

The linear form of equation (1), i.e. with *n*=1, is solved through implicit integration using the trapezoidal rule with some initial parameters that can be supplied by the user, yielding the first-order sample conversion. The non-linear form of equation (1) is then solved using Newtons-method with the calculated first-order conversion as the intital values, yielding the n-th order sample conversion *x_calc*. To avoid fluctuating behaivior, the calculated reaction rate -dm_calc/dt is then determined by finite difference scheme instead of equation (1) directly: 

    -dm_calc/dt = c*(x_calc[i+1] - x_calc[i])/dt                 (2)
  
where *dt* is the time step. An objective function defined by

    sum( (dm_calc/dt - dm_exp/dt)/max(dm_exp) )^2 )/n_points     (3)

where *n_points* are the number of data points. The objective function reflects the goodness of the fit and is minimized by some search algorithm by repeating the above mentioned procedure with some new parameters determined by the search algorithm. CMA-ES from the cma packaged is a good choice when large numbers of parameters are to be optimized. The best fitting parameters are then returned to the user. Note that the obtained parameters and the goodness of the fit depend alot on the initial conditions. Powell algorithm seems more suitable when initial paramters are close to local minima.

#### Usage:  
    python tga-kinetics.py
Remember to accept every single experimental dataset in the input tab before running an optimization, else the program doesn't run.

#### Required python packages:  
numpy (tested 1.18.5)  
scipy (tested 1.5.0)  
tkinter (tested 8.6)  
matplotlib (tested 3.2.2)  
pylightxl  
cma
