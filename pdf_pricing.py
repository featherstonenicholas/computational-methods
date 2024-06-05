
import numpy as np
import matplotlib.pyplot as plt
import time
"""Calculates the value of a european call by direct integration of the pdf of the underlying value at expiry

Returns:
        _type_: _description_
"""

# Set the initial parameters
S0 = 100
K = 120
r = 0.05
q = 0.02
sig = 0.5
T = 1.0

#step-size
dS = 0.1

# number of grid points
n = 13
N = 2**n



def logNormal(S,*args):    
    # model under consideration
    model = 'LogNormal'
    #args order is (r,q,sig,S0,T)
    r   = args[0]
    q = args[1]
    sig   = args[2]
    S0 = args[3]
    T   = args[4]
    f = np.exp(-0.5*((np.log(S/S0)-(r-q-sig**2/2)*T)/(sig*np.sqrt(T)))**2)/(sig*S*np.sqrt(2*np.pi*T))
    return f, model


def call_pdf(N,dS, pdf, *args):
    #start timing
    t0 = time.time()
    #create discount factor
    df = np.exp(-r*T)
    #create the S value mesh
    S=np.array([1.0+j*dS for j in range(N)])
    #evaluate the pdf across our mesh
    pdf_values, model=pdf(S, *args)
    #evaluate entrie integrand
    call=np.maximum(S-K,np.zeros(N))*pdf_values
    #perform numerical integration using trapezium
    c_0=df*np.sum(dS*(call[1:]+call[:N-1])/2)
    p_0 = c_0 + K * np.exp(-r * T) - S0*np.exp(-r*T) 
    run_time=time.time()-t0
    print('Model under consideration is %s' % model)
    print('Code runs in %s seconds' % run_time)
    print('Value of Call option is %f' % c_0)
    print('Value of Put option is %f' % p_0)
    return c_0, p_0,run_time, model

args=(r,q,sig,S0,T)
c_0 , p_0, runtime , model = call_pdf(N,dS, logNormal ,*args)
