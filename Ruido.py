# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:01:53 2026

@author: sere2
"""

import matplotlib.pyplot as plt
import numpy as np

#%%

def mi_sen(vmax, dc, ff, ph, nn, fs):    
    #vmax, dc,ff,ph tienen valor por defecto
    Ts=1/fs 
    tt=np.arange(nn)*Ts
    xx=dc+vmax*np.sin(2*np.pi*ff*tt + ph)
    return tt,xx

#Teorema de Nyquist: fs>=2f
fs=1000 #frecuencia de muestreo, cantidad de muestras que toma por seg
#fn=fs/2
#ws=2*np.pi*fs
Ts=1/fs #periodo de muestreo, tiempo entre muestras
N=1000
tt=np.arange(N)*Ts

t1, x1 = mi_sen(vmax=1, dc=0,ff=35, ph=0, nn=N, fs=fs)
t2, x2 = mi_sen(vmax=1, dc=0,ff=65, ph=0, nn=N, fs=fs)
t3, x3 = mi_sen(vmax=1, dc=0,ff=96, ph=0, nn=N, fs=fs)
tt,xx= mi_sen(vmax=np.sqrt(2), dc=0,ff=1, ph=0, nn=N, fs=fs)
#%%
#Ruido
#np.random.normal(mu,sigma,size)
rn=np.random.normal(0,1,N)

x5=x1+rn
plt.title("Generador de señal ")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.plot(t1,x1, label="ff=35")
plt.plot(t2,x2, label="ff=65")
plt.plot(t3,x3, label="ff=96")
#plt.plot(tt,x5, label="Señal+Ruido")
plt.legend()
plt.grid(True)
plt.show()

Pr=np.var(xx)
print(Pr)
SNR=15

Pr2=10**(-SNR/10)
print("Potencia en funcion de SNR:", Pr2)
rn2=np.random.normal(0,np.sqrt(Pr2),N)
plt.plot(tt,rn2)
plt.show()

x6=xx + rn2
    
    
##la cantidad de segundos muestreados depende del valor de N, si N=fs -->1seg, N=2fs-->2seg

#%%

from scipy import signal as sig
no=10
N=100
dd=np.zeros(N)
dd[no]=1
yy=sig.convolve(xx,dd)
plt.plot(yy)
plt.title("Convolución")
plt.show()

#%%
y2=(1/N)*sig.convolve(rn2,np.flip(rn2))
plt.plot(y2)
plt.title("Correlación")
plt.show()
#%%
#Cuantización
B=3
Vfs=3
qq=Vfs/2**B

xxq=np.round(x6/qq)
plt.plot(xxq)
plt.title("Cuantización")
plt.show()

#Error
# xxq*qq vuelvo a voltios
e=x6-(xxq*qq) 
plt.plot(e)
plt.title("Error de cuantización")
plt.legend()
plt.show()

#tiene que quedar entre -q/2 y q/2 (0.09375)

#%%



# Cuantización
B = 3
Vfs = 3
qq = Vfs / 2**B

xxq = np.round(x6 / qq)

# Volver a voltios
xxq_v = xxq * qq

# Gráfico comparación
plt.figure()
plt.plot(tt, x6, label="Señal original")
plt.step(tt, xxq_v, label="Señal cuantizada")
plt.title("Señal original vs cuantizada")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()


# Error
e = x6 - xxq_v

plt.figure()
plt.plot(tt, e, label=f"Error entre {-qq/2} y {qq/2}")
plt.title("Error de cuantización")
plt.xlabel("Tiempo")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

#%%

# Autocorrelación del error
e_corr = np.correlate(e, e, mode='full') / len(e)
lags = np.arange(-len(e)+1, len(e))

# Crear figura con 2 subplots
plt.figure(figsize=(10,5))

# 🔹 Histograma
plt.subplot(1,2,1)
plt.hist(e, bins=25, density=True)
plt.title("Histograma del error")
plt.xlabel("Error")
plt.ylabel("Amplitud")
plt.grid(True)

# 🔹 Autocorrelación
plt.subplot(1,2,2)
plt.plot(lags, e_corr)
plt.title("Autocorrelación del error")
#plt.xlabel("Lag")
plt.ylabel("Correlación")
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
#FFT
plt.figure()
XX=np.fft.fft(x6)
XXmod=np.abs(XX)
XXmod_dB=20*np.log(XXmod)
XXphase=np.angle(XX)
plt.subplot(2, 1, 1)
plt.title("FFT")
plt.plot(XXmod_dB, label="Modulo", color="red")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(XXphase, label="Fase")
plt.legend()
plt.show()


#como se ve el espectro de la señal de entrada? SNR?
SNRdB=10*np.log(1/Pr2)#--> potencia de señal normalizada
plt.plot(SNRdB)
#Densidad espectral de potencia |x|**2
#Ancho de banda Bw=nyquist=fs/2
#Pr=fs/2*rn-->rn=Pr/(fs/2)
#si aumenta Bw=f/s el piso de ruido disminuye

