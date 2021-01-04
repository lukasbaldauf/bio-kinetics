# bio-kinetics

**Bio-kinetics** estimates kinetic parameters from thermogravimetric data time, temperature and mass data. It aims tosimplify kinetic analysis of TGA data for its users through a graphical user interface.It was initially designed for biochar gasification but is also applicable to other TGA experiments. A set of simulated test-data experiments are supplied to test the program and illustrate the input-data structure. Excel files are also supported as input, given that the data structure matches tath of the test-data.


![Screenshot](https://github.com/lukasbaldauf/bio-kinetics/blob/main/program_screenshot.png)

The parameter estimation is based on the Arrhenius type rate equation:

    -dm/dt = c*dx/dt = A*exp(-E/(R * T))*x^n                     (1)

where dm/dt is the derivative of the normalized mass of the sample with respect to time, c is a scaling factor to convert the conversion-rate dx/dt to -dm/dt, A is the frequency-factor, E is the activation energy, R the gas constant, x is the conversion (i.e. the  fraction of sample left to react which is between 1 and 0) and n is the reaction order (can be fractional). 

The linear form of equation (1) (i.e. with n=1) is solved through implicit integration using the trapezoidal rule with some initial parameters that can be supplied by the user,yielding the first-order conversion. The non-linear form of equation (1) is then solved using Newtons-method with the first-order conversion as the intital values, yielding the n-th order sample conversion x_calc. To avoid fluctuating behaivior, the calculated reaction rate -dm_calc/dt is then determined by finite difference scheme: 

    -dm_calc/dt = c*(x_calc[i+1] - x_calc[i])/dt                 (2)
  
where dt is the time step. An objective function defined by

    sum( (dm_calc/dt - dm_exp/dt)/max(dm_exp) )^2 )/n_points     (3)

where n_points are the number of data points is calculated to yield the goodness of the fit. This objective function is minimized by some search algorithm (either Nelder-mead or Powell, others will be added later) by repeating the above mentioned procedure with some new parameters determined by the search algorithm. The best fit is the returned to the user. Note that the obtained parameters and the fit-goodness depends alot on the initial conditions.

#### Usage:  
       python bio-kinetics.py

#### Required python packages:  
numpy (tested 1.18.5)  
scipy (tested 1.5.0)  
tkinter (tested 8.6)  
matplotlib (tested 3.2.2)  
pylightxl  
