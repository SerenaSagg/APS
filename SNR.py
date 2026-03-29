# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:23:09 2026

@author: sere2
"""
import matplotlib.pyplot as plt
import numpy as np


def mi_sen(vmax, dc, ff, ph, nn, fs):    
    Ts=1/fs 
    tt=np.arange(nn)*Ts
    xx=dc+vmax*np.sin(2*np.pi*ff*tt + ph)
    return tt,xx

fs=2 #frecuencia de muestreo, cantidad de muestras que toma por seg
Ts=1/fs #periodo de muestreo, tiempo entre muestras
N=1000
tt=np.arange(N)*Ts
tt,xx= mi_sen(vmax=1, dc=0,ff=fs/N, ph=0, nn=N, fs=fs)

#SNRdB
#potencia de la señal Ps normalizada = 1
#Pv: potencia del ruido
SNR=10
Pv=10**(-SNR/10)
v=np.random.normal(0,1,N)
xv=xx + v

SNRdB=10*np.log(1/Pv)

plt.figure(1)

# Señal original
plt.subplot(3,1,1)
plt.plot(tt, xx)
plt.title("Señal original")

# Ruido
plt.subplot(3,1,2)
plt.plot(tt, v, color="orange")
plt.title("Ruido")

# Señal con ruido
plt.subplot(3,1,3)
plt.plot(tt, xv, color="green")
plt.title(f"Señal con ruido (SNR = {SNRdB:.2f} dB)")

plt.show()

#%%
#FFT
XX=np.fft.fft(xv)*2/N  #escalado por 1/N, y se multiplica por 2 para que se tenga en cuenta la otra mitad periodica
XXmod=np.abs(XX)
XXphase=np.angle(XX)
df=fs/N
ff=np.arange(N)*df
Pot=XXmod**2
Pot_dB=(10*np.log10(Pot))  
#Piso=Pv*2/fs
#Piso_dB=10*np.log10(Piso)
SNR_nuevo=10*np.log10(Pv)-10*np.log10(fs/2)

bBool=ff<=fs/2
plt.figure(2)
plt.subplot(2, 1, 1)
plt.title("FFT")
#plt.plot(ff[bBool],Pot_dB[bBool], label="Modulo", color="green")
plt.plot(ff[bBool],XXmod[bBool], label="Modulo", color="green")
#plt.axhline(Piso_dB, color="red")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.legend()
#plt.subplot(2, 1, 2)
#plt.plot(XXphase, label="Fase")
#plt.legend()
plt.show()


#punto max --> frecuencia donde hay mayor energia = ff

#Normalizar ancho de banda???