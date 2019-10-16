# Python code for CREW (Cognitive REceiver and Waveform design)

## Reference papers:
1. Stoica P., He H., Li J. [Optimization of the Receive Filter and Transmit Sequence for Active Sensing](https://ieeexplore.ieee.org/document/6104176), in IEEE Transactions on Signal Processing, vol. 60, no. 4, pp. 1730-1740, April 2012.
2. Soltanalian M., Tang B., Li J., Stoica P. [Joint Design of the Receive Filter and Transmit Sequence for Active Sensing](https://ieeexplore.ieee.org/document/6472022), in IEEE Signal Processing Letters, vol. 20, no. 5, pp. 423-426, May 2013.
3. MATLAB code written by Tang Bo

## Main Program of Cyclic Crew, which concludes:
### 1. Waveform and receiver design result by Cyclic CREW under both unimodular and PAR constraints for code length N = 100 and Power spectral density (PSD) of the optimized waveform and frequency response of the optimized receiver
![Hello](/figs/spectral_power_unimodular.png)
![Hello](/figs/freq_response_unimodular.png)

![Hello](/figs/spectral_power_par.png)
![Hello](/figs/freq_response_par.png)

### 2. Comparison with CAN-IV and CREW(Freq) for various code lengths
![Hello](/figs/mse_N.png)
![Hello](/figs/cpu_N.png)
