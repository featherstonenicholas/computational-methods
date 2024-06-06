import numpy as np
import cmath
import math
import time

# Fixed Parameters
S0 = 100
K = 80
k = math.log(K)
r = 0.05
q = 0.01

# Parameters for FFT 
n = 10
N = 2**n

#step-size
eta = 0.25
# damping factor
alpha = -1.5

# step-size in log strike space
lda = (2*math.pi/N)/eta

#Choice of beta
#beta = np.log(S0)-N*lda/2
beta = np.log(K)

#model-specific Parameters
model = 'VG'

params = []     
if (model == 'GBM'):
    
    sig = 0.3
    params.append(sig);
    
elif (model == 'VG'):
    
    sig = 0.3
    nu = 0.5
    theta = -0.4
    #
    params.append(sig);
    params.append(nu);
    params.append(theta);
    
elif (model == 'Heston'):
    
    kappa = 2.0
    theta = 0.05
    sig = 0.30
    rho = -0.70
    v0 = 0.04
    #
    params.append(kappa)
    params.append(theta)
    params.append(sig)
    params.append(rho)
    params.append(v0)

def generic_CF(u, params, S0, r, q, T, model):
    
    if (model == 'GBM'):
        
        sig = params[0]
        mu = np.log(S0) + (r-q-sig**2/2)*T
        a = sig*np.sqrt(T)
        phi = np.exp(1j*mu*u-(a*u)**2/2)
        
    elif(model == 'Heston'):
        
        kappa  = params[0]
        theta  = params[1]
        sigma  = params[2]
        rho    = params[3]
        v0     = params[4]
        
        tmp = (kappa-1j*rho*sigma*u)
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)
        
        pow1 = 2*kappa*theta/(sigma**2)
        
        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*math.log(S0)
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
        log_phi = numer1 - log_denum1 - tmp2
        phi = np.exp(log_phi)
        
        #g = np.sqrt((kappa-1j*rho*sigma*u)**2+(u*u+1j*u)*sigma*sigma)
        #beta = kappa-rho*sigma*1j*u
        #tmp = g*T/2
        
        #temp1 = 1j*(np.log(S0)+(r-q)*T)*u + kappa*theta*T*beta/(sigma*sigma)
        #temp2 = -(u*u+1j*u)*v0/(g/np.tanh(tmp)+beta)
        #temp3 = (2*kappa*theta/(sigma*sigma))*np.log(np.cosh(tmp)+(beta/g)*np.sinh(tmp))
        
        #phi = np.exp(temp1+temp2-temp3);
        

    elif (model == 'VG'):
        
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];

        if (nu == 0):
            mu = math.log(S0) + (r-q - theta -0.5*sigma**2)*T
            phi  = math.exp(1j*u*mu) * math.exp((1j*theta*u-0.5*sigma**2*u**2)*T)
        else:
            mu  = math.log(S0) + (r-q + math.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T
            phi = np.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu))

    return phi

def genericFFT(params, S0, K, r, q, T, alpha, eta, n, model):
    
    N = 2**n
    
    # step-size in log strike space
    lda = (2*np.pi/N)/eta
    
    #Choice of beta
    #beta = np.log(S0)-N*lda/2
    beta = np.log(K)
    
    # forming vector x and strikes km for m=1,...,N
    km = np.zeros((N))
    xX = np.zeros((N))
    
    # discount factor
    df = math.exp(-r*T)
    
    nuJ = np.arange(N)*eta
    psi_nuJ = generic_CF(nuJ-(alpha+1)*1j, params, S0, r, q, T, model)/((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))
    
    for j in range(N):  
        km[j] = beta+j*lda
        if j == 0:
            wJ = (eta/2)
        else:
            wJ = eta
        xX[j] = cmath.exp(-1j*beta*nuJ[j])*df*psi_nuJ[j]*wJ
     
    yY = np.fft.fft(xX)
    cT_km = np.zeros((N))  
    for i in range(N):
        multiplier = math.exp(-alpha*km[i])/math.pi
        cT_km[i] = multiplier*np.real(yY[i])
    
    return km, cT_km
print(' ')
print('===================')
print('Model is %s' % model)
print('-------------------')
    
T = 1
    
print(' ')
start_time = time.time()
km, cT_km = genericFFT(params, S0, K, r, q, T, alpha, eta, n, model)
if alpha>0:
    cT_k = cT_km[0]
    pT_k= cT_k + K * np.exp(-r * T) - S0*np.exp(-q*T) 
elif alpha<-1:
    pT_k=cT_km[0]
    cT_k= pT_k - K * np.exp(-r * T) + S0*np.exp(-q*T)
    

elapsed_time = time.time() - start_time
    
#cT_k = np.interp(np.log(), km, cT_km)
print("Option via FFT: for strike %s the call option premium is %6.4f" % (np.exp(k), cT_k))
print("Option via FFT: for strike %s the put option premium is %6.4f" % (np.exp(k), pT_k))
#print("Option via FFT: for strike %s the option premium is %6.4f" % (np.exp(k), cT_km[0]))
print('FFT execution time was %0.7f' % elapsed_time)