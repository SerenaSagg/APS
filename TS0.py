import matplotlib.pyplot as plt
import numpy as np

def mi_sen(vmax, dc, ff, ph, nn, fs):    
    #vmax, dc,ff,ph tienen valor por defecto
    Ts=1/fs 
    tt=np.arange(nn)*Ts
    xx=dc+vmax*np.sin(2*np.pi*ff*tt + ph)
    return tt,xx

#Teorema de Nyquist: fs>=2f
fs=100  #frecuencia de muestreo, cantidad de muestras que toma por seg
#fn=fs/2
#ws=2*np.pi*fs
#Ts=1/fs #periodo de muestreo, tiempo entre muestras
N=100

tt,xx= mi_sen(vmax=1, dc=0,ff=1, ph=0, nn=N, fs=fs)
tt,x1= mi_sen(vmax=1, dc=0,ff=50, ph=0, nn=N, fs=fs) #Nyquist
tt,x2= mi_sen(vmax=1, dc=0,ff=99, ph=0, nn=N, fs=fs)
tt,x3= mi_sen(vmax=1, dc=0,ff=101, ph=0, nn=N, fs=fs)
tt,x4= mi_sen(vmax=1, dc=0,ff=201, ph=0, nn=N, fs=fs)

#Cuando se pasa de fn la señal comienza a tener fa=fs-f

plt.title("Generador de señal ")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.plot(tt,xx, label="f=1 Hz")
plt.legend()
plt.grid(True)
plt.show()

plt.title("Generador de señal ")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.plot(tt,x1, label="f=50 Hz")
plt.plot(tt,x2, label="f=99 Hz")
plt.plot(tt,x3, label="f=101 Hz")
plt.plot(tt,x4, label="f=201 Hz")
plt.legend()
plt.grid(True)
plt.show()

    
    
##la cantidad de segundos muestreados depende del valor de N, si N=fs -->1seg, N=2fs-->2seg


