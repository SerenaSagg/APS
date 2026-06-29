#%%
##filtro IIR 
##Entrada->filtro iiR-> respuesta en frecuencia

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

wp = 70   #banda pasante
ws = 100   #banda de rechazo
gpass = 1 #dB maximo de perdida
gstop = 50 #dB minimo de atenuacion

fs = 500 #Al modificar fs, cambia la versión digital del filtro que diseña SciPy, por lo que puede cambiar su orden y sus coeficientes aunque wp y ws sigan siendo los mismos en Hz.
#cambia el eje de frecuencia , las frecuencias de interés (70 y 100 Hz) se van moviendo cada vez más hacia la izquierda cuando aumentás fs.

#ftype='butter'
#ftype='cauer'
ftype='cheby1'
#ftype='cheby2'
#b_coef coeficiente del numerador
#a_coef coeficiente del denominador
b_coef, a_coef = signal.iirdesign(wp, ws, gpass, gstop, analog= False, ftype= ftype, output= 'ba', fs= fs)

# los polos en radio unitario por estar en s (NO en z)

#al hacerlo en z (con analog false en lugar de true) me da de orden 3 en lugar de 4, 
#posiblemente por la compresion que genera el pasar de fs=500 a fs=2 pi
#al aumentar mi fs vuelvo a tener 4 polos ya que pasa a ser 'lineal' nuevamente 
#(porque deja de haber tanta compresion ya que la compresion esta muy lejos)
#lo que vimos de prewarping

taps = b_coef.shape[0] #cuantos coeficientes tiene el numerador

omega, resp_freq = signal.freqz(b_coef, a=a_coef, worN= 1024, fs=fs) #Respuesta en frecuencia normalizada, evalua en 1024 valores entre 0 y pi


fig, axs= plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(omega, 20*np.log10(abs(resp_freq)), 'C0', label="Modulo")
ax1.set_ylabel("Amplitude ", color='C0') #Modulo: cuánto amplifica o atenúa cada frecuencia
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi)) 
##El filtro deja pasar todas las frecuencias menos una cerca de 100Hz
#ax2 = ax1.twinx() #para que tengan mismo eje x


phase = np.unwrap(np.angle(resp_freq))
#phase = np.angle(resp_freq) #el de angle es el que solemos usar pero scipy recomendaba unwrap

ax2.plot(omega, phase, 'C1', label="Fase") #Fase: dice cuánto se retrasa cada frecuencia
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.legend()
plt.show()





#%%
##filtro IIR Butterworth pasabajos digital
##Entrada->filtro iiR-> respuesta en frecuencia

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

wp = 70   #banda pasante
ws = 100   #banda de rechazo
gpass = 1 #dB maximo de perdida
gstop = 50 #dB minimo de atenuacion

fs = 500 #Al modificar fs, cambia la versión digital del filtro que diseña SciPy, por lo que puede cambiar su orden y sus coeficientes aunque wp y ws sigan siendo los mismos en Hz.
#cambia el eje de frecuencia , las frecuencias de interés (70 y 100 Hz) se van moviendo cada vez más hacia la izquierda cuando aumentás fs.

#filgro iiR butter
#b_coef coeficiente del numerador
#a_coef coeficiente del denominador
b_coef, a_coef = signal.iirdesign(wp, ws, gpass, gstop, analog= False, ftype= 'butter', output= 'ba', fs= fs)

# los polos en radio unitario por estar en s (NO en z)

#al hacerlo en z (con analog false en lugar de true) me da de orden 3 en lugar de 4, 
#posiblemente por la compresion que genera el pasar de fs=500 a fs=2 pi
#al aumentar mi fs vuelvo a tener 4 polos ya que pasa a ser 'lineal' nuevamente 
#(porque deja de haber tanta compresion ya que la compresion esta muy lejos)
#lo que vimos de prewarping

taps = b_coef.shape[0] #cuantos coeficientes tiene el numerador

omega, resp_freq = signal.freqz(b_coef, a=a_coef, worN= 1024, fs=fs) #Respuesta en frecuencia normalizada, evalua en 1024 valores entre 0 y pi

fig, axs= plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(omega, abs(resp_freq), 'C0', label="Modulo")
ax1.set_ylabel("Amplitude ", color='C0') #Modulo: cuánto amplifica o atenúa cada frecuencia
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi)) 
#ax2 = ax1.twinx() #para que tengan mismo eje x
plt.legend()

phase = np.unwrap(np.angle(resp_freq))
#phase = np.angle(resp_freq) #el de angle es el que solemos usar pero scipy recomendaba unwrap

ax2.plot(omega, phase, 'C1', label="Fase") #Fase: dice cuánto se retrasa cada frecuencia
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.legend()
plt.show()

#%%
##filtro IIR CAUER pasabajos digital
##Entrada->filtro iiR-> respuesta en frecuencia

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

wp = 70   #banda pasante
ws = 100   #banda de rechazo
gpass = 3 #dB maximo de perdida
gstop = 10 #dB minimo de atenuacion

fs = 500 #Al modificar fs, cambia la versión digital del filtro que diseña SciPy, por lo que puede cambiar su orden y sus coeficientes aunque wp y ws sigan siendo los mismos en Hz.
#cambia el eje de frecuencia , las frecuencias de interés (70 y 100 Hz) se van moviendo cada vez más hacia la izquierda cuando aumentás fs.

#filgro iiR CAUER -> RECHAZABANDA (notch) -> tiene ondulaciones (Equiripple) en la banda de paso y en la banda de rechazo
#b_coef coeficiente del numerador
#a_coef coeficiente del denominador
b_coef, a_coef = signal.iirdesign(wp, ws, gpass, gstop, analog= False, ftype= 'cauer', output= 'ba', fs= fs)

# los polos en radio unitario por estar en s (NO en z)

#al hacerlo en z (con analog false en lugar de true) me da de orden 3 en lugar de 4, 
#posiblemente por la compresion que genera el pasar de fs=500 a fs=2 pi
#al aumentar mi fs vuelvo a tener 4 polos ya que pasa a ser 'lineal' nuevamente 
#(porque deja de haber tanta compresion ya que la compresion esta muy lejos)
#lo que vimos de prewarping

taps = b_coef.shape[0] #cuantos coeficientes tiene el numerador

omega, resp_freq = signal.freqz(b_coef, a=a_coef, worN= 1024, fs=fs) #Respuesta en frecuencia normalizada, evalua en 1024 valores entre 0 y pi


fig, axs= plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(omega, 20*np.log10(abs(resp_freq)), 'C0', label="Modulo")
ax1.set_ylabel("Amplitude ", color='C0') #Modulo: cuánto amplifica o atenúa cada frecuencia
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi)) 
##El filtro deja pasar todas las frecuencias menos una cerca de 100Hz
#ax2 = ax1.twinx() #para que tengan mismo eje x


phase = np.unwrap(np.angle(resp_freq))
#phase = np.angle(resp_freq) #el de angle es el que solemos usar pero scipy recomendaba unwrap

ax2.plot(omega, phase, 'C1', label="Fase") #Fase: dice cuánto se retrasa cada frecuencia
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.legend()
plt.show()

#%%
##filtro IIR CHEBY1 
##Entrada->filtro iiR-> respuesta en frecuencia

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

wp = 70   #banda pasante
ws = 100   #banda de rechazo
gpass = 3 #dB maximo de perdida
gstop = 10 #dB minimo de atenuacion

fs = 500 #Al modificar fs, cambia la versión digital del filtro que diseña SciPy, por lo que puede cambiar su orden y sus coeficientes aunque wp y ws sigan siendo los mismos en Hz.
#cambia el eje de frecuencia , las frecuencias de interés (70 y 100 Hz) se van moviendo cada vez más hacia la izquierda cuando aumentás fs.

#filgro iiR CHEBY1 -> tiene Equiripple (ondulaciones) en la banda de paso. 
#A medida que aumenta el orden de la funcion aumenta la contidad de oscilaciones.


#b_coef coeficiente del numerador
#a_coef coeficiente del denominador
b_coef, a_coef = signal.iirdesign(wp, ws, gpass, gstop, analog= False, ftype= 'cheby1', output= 'ba', fs= fs)

# los polos en radio unitario por estar en s (NO en z)

#al hacerlo en z (con analog false en lugar de true) me da de orden 3 en lugar de 4, 
#posiblemente por la compresion que genera el pasar de fs=500 a fs=2 pi
#al aumentar mi fs vuelvo a tener 4 polos ya que pasa a ser 'lineal' nuevamente 
#(porque deja de haber tanta compresion ya que la compresion esta muy lejos)
#lo que vimos de prewarping

taps = b_coef.shape[0] #cuantos coeficientes tiene el numerador

omega, resp_freq = signal.freqz(b_coef, a=a_coef, worN= 1024, fs=fs) #Respuesta en frecuencia normalizada, evalua en 1024 valores entre 0 y pi


fig, axs= plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(omega, 20*np.log10(abs(resp_freq)), 'C0', label="Modulo")
ax1.set_ylabel("Amplitude ", color='C0') #Modulo: cuánto amplifica o atenúa cada frecuencia
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi)) 
##El filtro deja pasar todas las frecuencias menos una cerca de 100Hz
#ax2 = ax1.twinx() #para que tengan mismo eje x


phase = np.unwrap(np.angle(resp_freq))
#phase = np.angle(resp_freq) #el de angle es el que solemos usar pero scipy recomendaba unwrap

ax2.plot(omega, phase, 'C1', label="Fase") #Fase: dice cuánto se retrasa cada frecuencia
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.legend()
plt.show()

#%%
##filtro IIR CHEBY2
##Entrada->filtro iiR-> respuesta en frecuencia

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

wp = 70   #banda pasante
ws = 100   #banda de rechazo
gpass = 3 #dB maximo de perdida
gstop = 10 #dB minimo de atenuacion

fs = 500 #Al modificar fs, cambia la versión digital del filtro que diseña SciPy, por lo que puede cambiar su orden y sus coeficientes aunque wp y ws sigan siendo los mismos en Hz.
#cambia el eje de frecuencia , las frecuencias de interés (70 y 100 Hz) se van moviendo cada vez más hacia la izquierda cuando aumentás fs.

#filgro iiR CHEBY2 -> tiene Equiripple (ondulaciones) en la banda de rechazo
#b_coef coeficiente del numerador
#a_coef coeficiente del denominador
b_coef, a_coef = signal.iirdesign(wp, ws, gpass, gstop, analog= False, ftype= 'cheby2', output= 'ba', fs= fs)

# los polos en radio unitario por estar en s (NO en z)

#al hacerlo en z (con analog false en lugar de true) me da de orden 3 en lugar de 4, 
#posiblemente por la compresion que genera el pasar de fs=500 a fs=2 pi
#al aumentar mi fs vuelvo a tener 4 polos ya que pasa a ser 'lineal' nuevamente 
#(porque deja de haber tanta compresion ya que la compresion esta muy lejos)
#lo que vimos de prewarping

taps = b_coef.shape[0] #cuantos coeficientes tiene el numerador

omega, resp_freq = signal.freqz(b_coef, a=a_coef, worN= 1024, fs=fs) #Respuesta en frecuencia normalizada, evalua en 1024 valores entre 0 y pi


fig, axs= plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(omega, 20*np.log10(abs(resp_freq)), 'C0', label="Modulo")
ax1.set_ylabel("Amplitude ", color='C0') #Modulo: cuánto amplifica o atenúa cada frecuencia
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi)) 
##El filtro deja pasar todas las frecuencias menos una cerca de 100Hz
#ax2 = ax1.twinx() #para que tengan mismo eje x


phase = np.unwrap(np.angle(resp_freq))
#phase = np.angle(resp_freq) #el de angle es el que solemos usar pero scipy recomendaba unwrap

ax2.plot(omega, phase, 'C1', label="Fase") #Fase: dice cuánto se retrasa cada frecuencia
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.legend()
plt.show()

#%%

##PLANTILLA PASABANDA
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

wp1 = 1   #banda pasante
ws1 = 0.1 #banda de rechazo
wp2= 35
ws2= 45
gpass = 1 #dB maximo de perdida
gstop = 40 #dB minimo de atenuacion
wp= [wp1, wp2]
ws= [ws1, ws2]

fs = 500 #Al modificar fs, cambia la versión digital del filtro que diseña SciPy, por lo que puede cambiar su orden y sus coeficientes aunque wp y ws sigan siendo los mismos en Hz.
#cambia el eje de frecuencia , las frecuencias de interés (70 y 100 Hz) se van moviendo cada vez más hacia la izquierda cuando aumentás fs.

ftype='butter'
#ftype='cauer'
#ftype='cheby1'
#ftype='cheby2'
#b_coef coeficiente del numerador
#a_coef coeficiente del denominador
b_coef, a_coef = signal.iirdesign(wp, ws, gpass, gstop, analog= False, ftype= ftype, output= 'ba', fs= fs)
sos_coef = signal.iirdesign(wp, ws, gpass, gstop, analog= False, ftype= ftype, output= 'sos', fs= fs)

# los polos en radio unitario por estar en s (NO en z)

#al hacerlo en z (con analog false en lugar de true) me da de orden 3 en lugar de 4, 
#posiblemente por la compresion que genera el pasar de fs=500 a fs=2 pi
#al aumentar mi fs vuelvo a tener 4 polos ya que pasa a ser 'lineal' nuevamente 
#(porque deja de haber tanta compresion ya que la compresion esta muy lejos)
#lo que vimos de prewarping

#taps = b_coef.shape[0] #cuantos coeficientes tiene el numerador
taps = sos_coef.shape[0]*2

#omega, resp_freq = signal.freqz(b_coef, a=a_coef, worN= 1024, fs=fs) #Respuesta en frecuencia normalizada, evalua en 1024 valores entre 0 y pi
omega, resp_freq = signal.freqz_sos(sos_coef, worN= 1024, fs=fs)

fig, axs= plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap IIR Filter")
#ax1.axvline(f_c, color='black', linestyle=':', linewidth=0.8)
ax1.plot(omega, 20*np.log10(abs(resp_freq)), 'C0', label="Modulo")
ax1.set_ylabel("Amplitude ", color='C0') #Modulo: cuánto amplifica o atenúa cada frecuencia
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi)) 
##El filtro deja pasar todas las frecuencias menos una cerca de 100Hz
#ax2 = ax1.twinx() #para que tengan mismo eje x


phase = np.unwrap(np.angle(resp_freq))
#phase = np.angle(resp_freq) #el de angle es el que solemos usar pero scipy recomendaba unwrap

ax2.plot(omega, phase, 'C1', label="Fase") #Fase: dice cuánto se retrasa cada frecuencia
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.legend()
plt.show()