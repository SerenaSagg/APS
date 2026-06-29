from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt

#freqz Compute the frequency response of a digital filter.

taps, f_c = 80, 1.0  # number of taps and cut-off frequency
b_coeffs = np.array([1,2,1])
a_coeffs = np.array([1,0,0.5**2])

taps=b_coeffs.shape[0]
omega, resp_freq = sig.freqz(b_coeffs, a = a_coeffs,worN=1024)   ##Respuest aen frecuencia


fig, ax1 = plt.subplots(tight_layout=True)
#Taps= cantidad de coeficientes
ax1.set_title(f"Frequency Response of {taps} tap FIR Filter" + f"($f_c={f_c}$ rad/sample)")
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude", color='C0')
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi))
ax2 = ax1.twinx()
#phase = np.unwrap(np.angle(resp_freq))
phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()                                                                                                                                                             