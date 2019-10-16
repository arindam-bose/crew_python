# Python code for CREW (Cognitive REceiver and Waveform design)

## Main Program of Cyclic Crew, which concludes:
### 1. Waveform and receiver design result by Cyclic CREW under both unimodular and PAR constraints for code length N = 100 and Power spectral density (PSD) of the optimized waveform and frequency response of the optimized receiver
![Spectral Power(dB) for unimodular constraint](/figs/spectral_power_unimodular.png)
![Frequency Response(dB) for unimodular constraint](/figs/freq_response_unimodular.png)

![Spectral Power(dB) for PAR constraint](/figs/spectral_power_par.png)
![Frequency Response(dB) for PAR constraint](/figs/freq_response_par.png)

### 2. Comparison with CAN-IV and CREW(Freq) for various code lengths
![MSE vs. N for unimodular constraint](/figs/mse_N_unimodular.png)
![CPU time vs. N for unimodular constraint](/figs/cpu_N_unimodular.png)

![MSE vs. N for PAR constraint](/figs/mse_N_par.png)
![CPU time vs. N for PAR constraint](/figs/cpu_N_par.png)

## Packages needed:
1. numpy
2. scipy.linalg
3. matplotlib.pyplot
4. sys
5. time

## Reference papers:
1. Stoica P., He H., Li J. [Optimization of the Receive Filter and Transmit Sequence for Active Sensing](https://ieeexplore.ieee.org/document/6104176), in IEEE Transactions on Signal Processing, vol. 60, no. 4, pp. 1730-1740, April 2012.
2. Soltanalian M., Tang B., Li J., Stoica P. [Joint Design of the Receive Filter and Transmit Sequence for Active Sensing](https://ieeexplore.ieee.org/document/6472022), in IEEE Signal Processing Letters, vol. 20, no. 5, pp. 423-426, May 2013.
3. MATLAB code written by Tang Bo
