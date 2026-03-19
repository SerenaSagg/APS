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
fs=100  #frecuencia de muestreo, cantidad de muestras que toma por seg
#fn=fs/2
#ws=2*np.pi*fs
Ts=1/fs #periodo de muestreo, tiempo entre muestras
N=100
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
SNR=20
print(SNR)
Pr2=10**(-SNR/10)
print("Potencia en funcion de SNR:", Pr2)
rn2=np.random.normal(0,np.sqrt(Pr2),N)
plt.plot(tt,rn2)
plt.show()
    
    
##la cantidad de segundos muestreados depende del valor de N, si N=fs -->1seg, N=2fs-->2seg

#%%

from scipy import signal as sig
no=10
N=100
dd=np.zeros(N)
dd[no]=1
yy=sig.convolve(xx,dd)

#%%
y2=(1/N)*sig.convolve(rn2,np.flip(rn2))

#%%

B=4
Vfs=3
qq=Vfs/2**B

xxq=np.round(xx/qq)
