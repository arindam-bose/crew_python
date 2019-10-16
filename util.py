#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:06:30 2019

@author: Arindam
"""
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
import sys
import time

##############################################################################
def cyclic_crew(s_initial, gamma, beta, rho, epsilon=1e-4, epsilon2=1e-6, dlf = 'fro'):
    # Cyclic CREW Algorithm for Joint Design of the Receive Filter and Transmit Sequence for Active Sensing
    # usage: [s, w, mse] = cyclic_crew(s_init, gamma, beta, rho, {epsilon}, {epsilon2}, {dlf})
    # inputs:
    #       s_init:   initial Code of Cyclic CREW, Nx1, N is the code length
    #                 e.g., s_init can be random unimodular code, 
    #                 s_init = exp(1j*1000*rand(N,1)) or an existing unimodular code.
    #       gamma:    covariance matrix of the Interference, NxN
    #       beta:     average power of the clutter, 1x1
    #       rho:      peak to average ratio, 1x1, 1<=PAR<= N; 1=Unimodular, N=Energy constraint
    #       epsilon:  (optional) threshold to stop Cyclic CREW, 1x1, e.g., 1e-6
    #       epsilon2: (optional) threshold to stop the power-like method, 1x1, e.g., 1e-6
    #       dlf:      (optional) diagonal loading factor for making the Hermitian matrix in the quadratic 
    #                 optimization problem in the power-like method positive semidefinite, if
    #                 it is 'fro', we choose the Frobenius norm as the diagonal loading factor, another 
    #                 option is to use 'eigen', which will use the maximum eigen value.
    # outputs: 
    #       s:        the optimized waveform, Nx1
    #       w:        the optimized receiver, Nx1
    #       mse:      the MSE of the optimized waveform, 1x1
    # references:
    #       [1] Stoica P., He H., Li J. Optimization of the Receive Filter and Transmit Sequence for 
    #           Active Sensing, in IEEE Transactions on Signal Processing, vol. 60, no. 4, pp. 1730-1740, April 2012.
    #       [2] Soltanalian M., Tang B., Li J., Stoica P. Joint Design of the Receive Filter and Transmit Sequence 
    #           for Active Sensing, in IEEE Signal Processing Letters, vol. 20, no. 5, pp. 423-426, May 2013.
    #       [3] MATLAB code written by Tang Bo
    
    N = len(s_initial)       # Code Length
    s_pre = s_initial        # Previous Waveform
    
    # Euclidean distance between the MSE of the current waveform and the previous waveform
    err = lin.norm(s_pre)
    while (err > epsilon):
        ## Step 1: Optimize w with respect to (w.r.t.) a given s
        # Compute R  = gamma + beta*sum_{k=-N+1,k \neq 0}^{N-1}(J_k s s'*J_k');
        R = gamma + beta*compute_R(s_pre)
        # Compute Optimal receiver w.r.t. s
        w = lin.inv(R) @ s_pre
        
        
        ## Step 2: Optimize s w.r.t a given w under a PAR constraint
        # Compute the object value with w and s_pre
        obj_curr = (np.abs(w.conj().reshape(1,N) @ R @ w.reshape(N,1)) / 
                    (np.abs(w.conj().reshape(1,N) @ s_pre.reshape(N,1))**2)).item()
        
        # The idea of optimizing s w.r.t. w is based on fractional programming and UQP 
        # The equivalent optimization problem can be written as
        # s = argmin_{s} s'*[(w'*gamma*w)/N*eye(N) + beta*sum_{k=-N+1,k \neq 0}^{N-1}(J_k w w'*J_k')
        #                                           - obj_curr*w*w']*s,
        # which is a quadratic optimization problem

        # Compute the Hermitian matrix in the quadratic optimization problem
        R_UQP = ((w.conj().reshape(1,N) @ gamma @ w.reshape(N,1)) * np.identity(N)/N + 
                 beta*compute_R(w))
        R_UQP2 = R_UQP
        R_UQP = R_UQP - obj_curr*(w.reshape(N,1) @ w.conj().reshape(1,N))
        
        # Note that the minimization of s'*R_UQP*s is equivalent to the maxmimization of s'*(-R_UQP)*s
        # make R_UQP positive semidefinite by diagonal loading
        if (dlf == 'fro'):
            R_UQP = -R_UQP
            R_UQP = R_UQP + lin.norm(R_UQP,'fro')*np.identity(N)
        elif (dlf == 'eigen'):
            # maximum eigenvalue
            [eigR,_] = lin.eig(R_UQP)
            Dmax = np.max(np.abs(eigR))
            if (Dmax>0):
                R_UQP = (Dmax+0.1)*np.identity(N) - R_UQP
            else:
                R_UQP = -R_UQP
        
        # Optimize s with the power-like method
        err2 = 100
        while (err2 > epsilon2):
            # s = np.exp(1j*np.angle(R_UQP @ s_pre.reshape(N,1)))
            s_temp = (R_UQP @ s_pre.reshape(N,1)).ravel()
            
            # find the nearest vector to s_temp under PAR constraint
            s = nearest_vector(s_temp,rho)
            err2 = lin.norm(s_pre - s)

            s_pre = s # update current waveform
        
        # updated the MSE with the optimized waveform
        obj_upd = (np.abs(s.conj().reshape(1,N) @ R_UQP2 @ s.reshape(N,1)) /
                   (np.abs(w.conj().reshape(1,N) @ s.reshape(N,1))**2)).item()
        
        err = lin.norm(obj_upd - obj_curr)
        s_pre = s
    
    # Final Outputs: the optimized waveform, receiver and the MSE of the optimized waveform
    return s, w, obj_upd

##############################################################################
def compute_R(s):
    # Compute R in Equation (24) or Q in Equation (27)
    # namely, R = sum_{k=-(-N+1),k\neq0}^{k=N-1} J_k*s*s^{H}*J_k^{H}
    # usage: R = compute_R(s)
    # inputs:
    #       s: current Waveform or Current receiver weight, Nx1, N is the vector length
    # outputs:
    #       R: as R in Equation (24) or Q in Equation (27)
    
    N = len(s)
    R = np.zeros([N,N])
    R = R - (s.reshape(N,1) @ s.conj().reshape(1,N))
    scorr = np.correlate(s,s,'full')
    R = R + lin.toeplitz(scorr[N-1:], scorr[N-1:].conj())
    return R

##############################################################################
def nearest_vector(s, rho):
    # Finds the nearest vector to s under the PAR constraint
    # usage: x = nearest_vector(s, rho)
    # inputs:
    #       s:   vector, Nx1
    #       rho: Peak to average power ratio (PAR), PAR of a vector x is defined by 
    #            PAR(x) = max( |x_i|^2 )/( sum( |x_i|^2 )/N ), 1x1
    #            1<=rho<=N, 1=Unimodular, N=Energy constraint
    # outputs:
    #       x:   The nearest vector, Nx1
    
    if (rho < 0):
        print('rho should be larger than 0')
        sys.exit(0)
    # vector length
    N = len(s)

    # unimodular constraint
    if (rho == 1):
        x = np.exp(1j*np.angle(s))
        return x
    
    x = np.zeros(N, dtype=complex)
    x_temp = np.sqrt(N)*s/lin.norm(s)
    vec = np.arange(N)
    
    while (np.max(np.abs(x_temp)) > np.sqrt(rho)):
        # maximum of x_temp and its index
        I = np.argmax(np.abs(x_temp))
        x[vec[I]] = np.sqrt(rho) * np.exp(1j*np.angle(x_temp[I]))
        
        # delete the Ith element from s 
        s_list = s.tolist()
        vec_list = vec.tolist()
        del s_list[I]
        del vec_list[I]
        s = np.asarray(s_list)
        vec = np.asarray(vec_list)
        
        # Update N
        N = N - rho
        x_temp = np.sqrt(N)*s/lin.norm(s)
    
    x[vec] = x_temp
    return x

##############################################################################
def can_siso_par(rho, N, x0=None):
    # CAN algorithm with the PAR (peak-to-average ratio) constraint
    # usage: x = can_siso_par(rho, N, {x0})
    # inputs: 
    #       rho: PAR(x) = max{|x(n)|^2} <= rho (average power is 1), 1x1
    #       N: length of the sequence
    #       x0: the initialization sequence, Nx1
    # outputs: 
    #       x: the generated sequence, Nx1
    
    if (x0.all()==None):
        x = np.exp(1j*2*np.pi*np.random.rand(N))
    else:
        x = x0
        
    xPre = np.zeros(N)
    iterDiff = lin.norm(x - xPre)
    while (iterDiff > 1e-3):
        # print(iterDiff)
        xPre = x
        
        # step 2
        z = np.r_[x, np.zeros(N)] # 2Nx1
        f = 1/np.sqrt(2*N) * np.fft.fft(z)  # 2Nx1
        v = np.sqrt(1/2) * np.exp(1j * np.angle(f)) # 2Nx1
        
        # step 1
        g = np.sqrt(2*N) * np.fft.ifft(v) # 2Nx1    
        x = vector_fit_par(g[:N], N, rho) # Nx1
        
        # stop criterion
        iterDiff = lin.norm(x - xPre)
    return x

##############################################################################
def vector_fit_par(z, power, rho):
    # Tropp's Alternating Projection algorithm to compute the nearest vector
    # under the PAR (peak-to-average power ratio) constraint
    # PAR(s) = max{|s(n)|^2}/(power/N) <= rho
    # usage: s = vector_fit_par(z, power, rho)
    # inputs:
    #       z:     Nx1 vector, could be complex-valued
    #       power: the squared norm of s, i.e. ||s||^2
    #       rho:   the PAR of s <= rho
    # outputs:
    #       s:     Nx1 vector, obtained by sovling the following min problem:
    #              min_s || s-z ||
    #              s.t. PAR(s) <= rho
    #              ||s||^2 = power
    
    if (rho == 1):
        s = np.exp(1j * np.angle(z))
        s = s/lin.norm(s) * np.sqrt(power)
        return s
    
    N = len(z)
    delta = np.sqrt(power*rho/N)      # max{|s(n)}} <= delta
    index = np.argsort(np.abs(z))
    s = np.zeros(N, dtype=complex)
    
    for k in range(N,0,-1):         # {k components of z with smallest magnitude} denoted as II
        ind = index[:k]
        if ~np.any(z[ind]): # if all elements in II are zero
            s[ind] = np.sqrt((power - (N-k) * delta**2) / k)
            break
        else:
            gamma = np.sqrt((power - (N-k) * delta**2) / np.sum((np.abs(z[ind]))**2))
            sTmp = gamma * z[ind]
            if np.all((np.abs(sTmp)-(1e-7)) <= delta):  # satisfying the power constraint
                # 1e-7 is introduced to prevent numerical errors
                s[ind] = sTmp
                break
    
    # Besides II, the other N-k components
    ind = index[k:N]
    s[ind] = delta * np.exp(1j * np.angle(z[ind]))
    return s
    
############################################################################## 
def freq_crew(N, gamma, beta, rho=None, epsilon=1e-6, s0=None):
    # Frequency domain Lagrange plus cyclic algorithm for probing sequence and receive filter design
    # usage: [s, w, mse, msebound] = cyc_freq(N, gamma, beta, {rho}, {s0})
    # inputs:
    #       N:     N is the code length
    #       gamma: the covariance matrix of interference (noise+jamming), NxN
    #       beta:  E{|alpha|^2} where alpha is the RCS coefficient
    #       rho:   (optional) maximum allowed peak-to-average power ratio, in [1 N]
    #       epsilon:  (optional) threshold to stop freCREW, 1x1, e.g., 1e-4
    #       s0:    (optional) sequence for initialization Nx1
    #
    # outputs:
    #       s       : the probing sequence, Nx1
    #       w       : the receive filter, w = inv(P) * s, Nx1
    #       mse     : w'Rw/(|w's|^2) = 1/(s'*inv(R)*s)
    #       msebound: (2N-1)/sum(z ./ (beta * z + g)) - beta, the MSE when 
    #                 the power spectrum (i.e., z) of s is ideal
    
    if (rho==None): rho = N
    if (s0.all()==None):
        s = np.exp(1j*2*np.pi*np.random.rand(N))
    else:
        s = s0
        pass
    
    cn = gamma[:,0]  # covariance of noise+interference
    cn = np.r_[cn, np.flipud(cn[1:N]).conj()]  # (2N-1) x 1
    
    g = np.fft.fft(cn) / (2*N-1)   # power spectrum of noise+interference, (2N-1) x 1
    g = np.real(g)    # eliminate numerical problems
    rh = (beta * N + np.sum(g)) / (np.sum(np.sqrt(g)))
    z = (rh * np.sqrt(g) - g) / beta # (2N-1) x 1
    
    if (np.sum(z >= 0) < (2*N-1)):
        # binary search for lambda
        left = 0
        right = 1
        fval = 0
        while (np.abs(fval - N) > 1e-2):
            lamda = (left + right) / 2
            z = (lamda * rh * np.sqrt(g) - g) / beta   # (2N-1) x 1
            fval = np.sum(np.amax(np.c_[z, np.zeros(2*N-1)], axis=1))
            if (fval < N):
                left = lamda
            elif (fval > N):
                right = lamda
        z = np.amax(np.c_[z, np.zeros(2*N-1)], axis=1)
    
    # if the spectrum match is perfect
    msebound = (2*N-1) / np.sum(z/(beta*z + g)) - beta
#    print('MSE lower bound: %.6f' %msebound)
    
    # calculate {|x_p|}
    xabs = np.sqrt(z) # 2Nx1, {|x_p|}, p=1,...,2N-1
    
#    # examine the optimal sequence/filter spectrum
#    habs = xabs / (beta * z + g)
#    plt.figure()
#    plt.plot(np.linspace(0,1,2*N-1), z, 'r', label='optimal s')
#    plt.plot(np.linspace(0,1,2*N-1), habs**2, 'b', label='optimal w')
#    plt.legend()
#    plt.xlabel('f')
#    plt.ylabel('Power Spectrum')
#    plt.autoscale(enable=True, axis='x', tight=True)
#    plt.show()
    
    # iteration
    s_pre = np.zeros(N)
    iterDiff = lin.norm(s - s_pre)
    sptrmfit_pre = 100
    
    while (iterDiff > epsilon):
#        s_pre = s
        
        # update x
        x = np.multiply(xabs, np.exp(1j * np.angle(np.fft.fft(s, 2*N-1)))) # (2N-1) x 1

        # update s
        nu = np.fft.ifft(x) * np.sqrt(2*N-1) # (2N-1) x 1
        nu = nu[:N]   # Nx1    
        if (rho == 1):
            s = np.exp(1j * np.angle(nu))
        elif (rho < N):
            s = vector_fit_par(nu, N, rho)
        else:
            s = nu / lin.norm(nu) * np.sqrt(N)
        
        R = gamma + beta*compute_R(s)
        mse = 1/np.real(s.conj() @ lin.inv(R) @ s)
        sptrmfit = lin.norm(xabs - np.abs(np.fft.fft(s, 2*N-1)/np.sqrt(2*N-1)))**2
        
        # iteration criterion
        iterDiff = lin.norm(sptrmfit - sptrmfit_pre)
        sptrmfit_pre = sptrmfit
        
#        print('||s-spre|| = %.4f; MSE = %.4f; || |x| - |F\'s| || = %.4f' %(iterDiff, mse, sptrmfit))

    R = gamma + beta*compute_R(s)
    w = lin.inv(R) @ s
    mse = 1/np.real(s.conj() @ w)
    return s, w, mse, msebound

##############################################################################
def tic():
    # Generator that returns time differences, similar to MATLAB tic and toc functionality
    # usage: 
    #       tic()
    #       <do some stuff>
    #       t = toc(True)   # will return the elapsed time and print the statement
    #       t = toc(False)  # will return the elapsed time and NOT print the statement
    # references:
    #       [1] https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions, 
    #           answered by GuestPoster
    
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(tempBool=True):
    if 'startTime_for_tictoc' in globals():
        temp_time_interval = time.time() - startTime_for_tictoc
        if tempBool:
            print("Elapsed time is %.6f seconds.\n" %temp_time_interval)
        return temp_time_interval
    else:
        print ("Toc: start time not set")

##############################################################################  