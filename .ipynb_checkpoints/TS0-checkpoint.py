import matplotlib.pyplot as plt
import numpy as np

def mi_sen(vmax, dc, ff, ph, nn, fs):    
    #vmax, dc,ff,ph tienen valor por defecto
    xx=dc+vmax*np.sin(2*np.pi*ff*tt + ph)
    return tt,xx

#Teorema de Nyquist: fs>=2f
fs=100  #frecuencia de muestreo, cantidad de muestras que toma por seg
#fn=fs/2
#ws=2*np.pi*fs
Ts=1/fs #periodo de muestreo, tiempo entre muestras
N=100
tt=np.linspace(0,N-1)*Ts
6
t1, x1 = mi_sen(vmax=1, dc=0,ff=1, ph=0, nn=N, fs=fs)

plt.title("Generador de señal ")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
#plt.legend()
plt.grid(True)
plt.plot(t1,x1)
plt.show
    
    
##la cantidad de segundos muestreados depende del valor de N, si N=fs -->1seg, N=2fs-->2seg


