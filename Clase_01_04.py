# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:12:17 2026

@author: sere2
"""

import matplotlib.pyplot as plt
import numpy as np

nn=np.arange(0,8)
N=8
xx=4+3*np.sin((np.pi/2)*nn)
XX=np.fft.fft(xx)
XXmod=np.abs(XX)
XXphase=np.angle(XX)

Pot=(XXmod**2)*1/(N**2)

plt.subplot(2,1,1)
plt.title("Señal con pi/2")
plt.stem(nn,XXmod, label="Modulo", linefmt='slateblue', basefmt="royalblue")
plt.legend()
plt.grid()


plt.subplot(2,1,2)
plt.stem(nn,XXphase, label="Fase", linefmt='slateblue', basefmt="royalblue")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.stem(nn,Pot, linefmt='slateblue', basefmt="royalblue")
plt.title("Espectro de potencia con pi/2")
plt.grid()
plt.show()

##Cambio de fase donde no cae sobre un bin, la energia se divide entre los bins
#Desparramo espectral
xx=4+3*np.sin((np.pi/3)*nn)
XX=np.fft.fft(xx)
XXmod=np.abs(XX)
XXphase=np.angle(XX)

Pot=(XXmod**2)*1/(N**2)

plt.figure()
plt.subplot(2,1,1)
plt.title("Señal con pi/3")
plt.stem(nn,XXmod, label="Modulo", linefmt='slateblue', basefmt="royalblue")
plt.legend()
plt.grid()


plt.subplot(2,1,2)
plt.stem(nn,XXphase, label="Fase", linefmt='slateblue', basefmt="royalblue")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.stem(nn,Pot, linefmt='slateblue', basefmt="royalblue")
plt.title("Espectro de potencia con pi/3")
plt.grid()
plt.show()

##si quiero ver una frecuencia pi/3, cuantas muestras necesito?
##La cantidad de muestras N tal que 2pi/N=pi/3