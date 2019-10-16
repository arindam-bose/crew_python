#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:07:00 2019

@author: Arindam
"""

#   Main Program of Cyclic Crew, which concludes:
#  1. Waveform and receiver design result by Cyclic CREW under both unimodular and PAR constraints for code length N = 100  
#  2. Power spectral density (PSD) of the optimized waveform and frequency response of the optimized receiver
#  3. Comparison with CAN-IV and CREW(Freq) for various code lengths

import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from util import *

# Simulation parameter
N = 100       # code Length
sigmaJ = 100  # jammer power
sigmaN = 0.1  # noise power, 30dB jammer to noise power ratio (JNR)
beta = 1      # average power of clutter

# Jamming type: 1-Spot jamming;  2-Barrage jamming 
# If the lower band of jamming frequency equals the upper band, then Spot Jamming, Otherwise barrage jamming
#  normalized frequency band of the jammer
fre_lb = 0.2  #  lower band
fre_ub = 0.3  #  upper band

# Compute jamming power spectral density (PSD)
eta = np.zeros([2*N-1])
eta[np.int(fre_lb*(2*N-1)):np.int(fre_ub*(2*N))] = 1

# Correlation coefficient of jammer
qcorr = np.fft.ifft(eta)

# jammer covariance matrix
vector1 = qcorr[:N] / qcorr[0]
vector2 = np.r_[qcorr[0], np.flipud(qcorr[N:])] / qcorr[0]
gammaJ = lin.toeplitz(vector1, vector2)  #  \gamma_J in equation (18)

# jammer plus noise covariance matrix
gamma = sigmaJ*gammaJ + sigmaN*np.identity(N)

# correlation coefficient of jammer plus noise
gamma_corr = np.r_[gamma[0:N,0], np.flipud(gamma[0,1:N]).T]

# power spectral density of the jammer plus noise
spectrum = np.abs(np.fft.fft(gamma_corr))/len(gamma_corr)

# Initial code ---- Golumb code
s_initial = np.exp(np.multiply(1j*np.pi*np.arange(1,N+1), np.arange(0,N)/N))


##############################################################################
# Unimodular waveform design
rho = 1              # PAR = 1
epsilon_main = 1e-6  # Threshold to stop Cyclic CREW
epsilon_inner = 1e-6 # Threshold to stop the power-like method

s_cycCREW_unimodular, w_cycCREW_unimodular, mse_cycCREW_unimodular = cyclic_crew(
        s_initial, gamma, beta, rho, epsilon_main, epsilon_inner)

# Power spectral density of the optimized waveform
PSD_waveform_unimodular = np.fft.fft(s_cycCREW_unimodular, 2*N)

# Frequency Response of the optimized receiver
freq_receiver_unimodular = np.fft.fft(w_cycCREW_unimodular, 2*N)

# Plot for spectral power(dB) vs. normalized frequency
plt.figure()
plt.plot(np.arange(2*N)/(2*N), 20*np.log10(np.abs(PSD_waveform_unimodular)/np.max(np.abs(PSD_waveform_unimodular))))
plt.xlabel('Normalized Frequency')
plt.ylabel('Spectral Power(dB)')
plt.title('Cyclic CREW--Unimodular Constraint')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

# Plot for frequency response(dB) vs. normalized frequency
plt.figure()
plt.plot(np.arange(2*N)/(2*N), 20*np.log10(np.abs(freq_receiver_unimodular)/np.max(np.abs(freq_receiver_unimodular))))
plt.xlabel('Normalized Frequency')
plt.ylabel('Frequency Response(dB)')
plt.title('Cyclic CREW--Unimodular Constraint')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

print('The MSE of the optimized waveform under the unimodular constraint is %.6f' %(mse_cycCREW_unimodular))


##############################################################################
# Waveform design under the PAR constraint
rho = 2              #  PAR = 2
epsilon_main = 1e-4  #  Threshold to stop Cyclic CREW
epsilon_inner = 1e-6 # Threshold to stop the power-like method

[s_cycCREW_PAR, w_cycCREW_PAR, mse_cycCREW_PAR] = cyclic_crew(
        s_initial, gamma, beta, rho, epsilon_main, epsilon_inner, 'eigen')

# Power spectral density of the optimized waveform
PSD_waveform_PAR = np.fft.fft(s_cycCREW_PAR, 2*N)

# Frequency Response of the optimized receiver
freq_Receiver_PAR = np.fft.fft(w_cycCREW_PAR, 2*N)

# Plot for spectral power(dB) vs. normalized frequency
plt.figure()
plt.plot(np.arange(2*N)/(2*N), 20*np.log10(np.abs(PSD_waveform_PAR)/np.max(np.abs(PSD_waveform_PAR))))
plt.xlabel('Normalized Frequency')
plt.ylabel('Spectral Power(dB)')
plt.title('Cyclic CREW--PAR Constraint')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

# Plot for frequency response(dB) vs. normalized frequency
plt.figure()
plt.plot(np.arange(2*N)/(2*N), 20*np.log10(np.abs(freq_Receiver_PAR)/np.max(np.abs(freq_Receiver_PAR))))
plt.xlabel('Normalized Frequency')
plt.ylabel('Frequency Response(dB)')
plt.title('Cyclic CREW--PAR Constraint')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

print('The MSE of the optimized waveform under the PAR constraint is %.6f' %(mse_cycCREW_PAR))


##############################################################################
# Comparison with other methods, including CAN-IV (i.e., CAN-MMF) and CREW(fre)
# Sequence length
N_set = np.array([25, 50, 100, 200, 500])
num = len(N_set)
sigmaJ = 100  # jammer power
sigmaN = 0.1  # noise power, 30dB jammer to noise power ratio (JNR)
beta = 1      # average power of clutter

# Jamming type: 1-Spot jamming;  2-Barrage jamming 
# If the lower band of jamming frequency equals the upper band, then Spot Jamming, Otherwise barrage jamming
#  normalized frequency band of the jammer
fre_lb = 0.2  #  lower band
fre_ub = 0.3  #  upper band

# mean square error(MSE)
mse_CAN = np.zeros(num)
mse_freCREW = np.zeros(num)
mse_cycCREW = np.zeros(num)
mse_bound = np.zeros(num)

# CPU time (similar to using MATLAB tic and toc function)
time_CAN  = np.zeros(num)
time_freCREW = np.zeros(num)
time_cycCREW = np.zeros(num)

rho = 2               #  for unimodular: 1, for PAR: 2
epsilon_main = 1e-4   #  Threshold to stop Cyclic CREW
epsilon_inner = 1e-6  #  Threshold to stop the power-like method

for k in range(num):
    
    N = N_set[k]
    print('N =', N)
    
    # power spectral density of the jammer
    eta = np.zeros([2*N-1])
    eta[np.int(fre_lb*(2*N-1)):np.int(fre_ub*(2*N))] = 1
    
    # Correlation coefficient of jammer
    qcorr = np.fft.ifft(eta)
    
    # jammer covariance matrix
    vector1 = qcorr[:N] / qcorr[0]
    vector2 = np.r_[qcorr[0], np.flipud(qcorr[N:])] / qcorr[0]
    gammaJ = lin.toeplitz(vector1, vector2)  #  \gamma_J in equation (18)
    
    # jammer plus noise covariance matrix
    gamma = sigmaJ*gammaJ + sigmaN*np.identity(N)
    
    # Initial code ---- Golumb code
    s_initial = np.exp(np.multiply(1j*np.pi*np.arange(1,N+1), np.arange(0,N)/N))
    
    # CAN-IV
    tic()
    s_CAN = can_siso_par(rho, N, s_initial)
    R = gamma + beta*compute_R(s_CAN)
    w_CAN = lin.inv(R) @ s_CAN
    time_CAN[k] = toc(False)
    mse_CAN[k] = 1/np.abs(w_CAN.conj() @ s_CAN)
    
    
    # CREW(Freq)
    tic()
    [s_freCREW, w_freCREW, mse_freCREW[k], mse_bound[k]] = freq_crew(N, gamma, beta, rho, epsilon_main, s_initial)
    time_freCREW[k] = toc(False)
    
    
    # CREW(Direct)
    tic()
    [s_cycCREW, w_cycCREW, mse_cycCREW[k]] = cyclic_crew(s_initial, gamma, beta, rho, epsilon_main, epsilon_inner, 'eigen')
    time_cycCREW[k] = toc(False)
    
# Plot for MSE vs. N
plt.figure()
plt.semilogy(N_set, mse_CAN, '-r*', label='CAN-MMF')
plt.semilogy(N_set, mse_freCREW, '-bd', label='CREW(fre)')
plt.semilogy(N_set, mse_cycCREW, '--b', label='CREW(cyclic)')
plt.xlabel('N')
plt.ylabel('MSE')
plt.title('MSE vs. N (Barrage jamming, PAR=%d)'%rho)
plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

# Plot for CPU time vs. N
plt.figure()
plt.semilogy(N_set, time_CAN, '-r*', label='CAN-MMF')
plt.semilogy(N_set, time_freCREW, '-bd', label='CREW(fre)')
plt.semilogy(N_set, time_cycCREW, '--b', label='CREW(cyclic)')
plt.xlabel('N')
plt.ylabel('time (sec)')
plt.title('CPU time vs. N (Barrage jamming, PAR=%d)'%rho)
plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()