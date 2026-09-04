# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:31:02 2026

@author: sere2
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def mi_sen(vmax, dc, ff, ph, nn, fs):
    Ts = 1/fs
    tt = np.arange(nn) * Ts
    xx = dc + vmax * np.sin(2*np.pi*ff*tt + ph)
    return tt, xx

def calcular_fft(x, fs):
    N = len(x)
    X = np.fft.fft(x)* 2/N
    df=fs/N
    ff=np.arange(N)*df
    modulo = np.abs(X)

    return ff, modulo

# Parámetros
Np = 10
fs = 20000
ff = fs/Np
N = 1000
T = 1/ff
tt_ruido = np.arange(N) / fs

# 1) Señal original de 2 kHz
tt1, x1 = mi_sen(vmax=1, dc=0, ff=ff, ph=0, nn=N, fs=fs)
P1=np.var(x1)
f1, X1 = calcular_fft(x1, fs)

# 2) Misma señal con potencia 2W y desfasada en pi/2
Pot=2
vmax=np.sqrt(Pot*2)
tt2, x2 = mi_sen(vmax=vmax, dc=0, ff=ff, ph=np.pi/2, nn=N, fs=fs)
P2=np.var(x2)
f2, X2 = calcular_fft(x2, fs)

# 3) Ruido aleatorio con distribucion normal
Var=0.1
nn1=np.random.normal(0,np.sqrt(Var),N)  # Ruido gaussiano de media 0 y varianza 0.1
f3, X3 = calcular_fft(nn1, fs)

# 4) Ruido aleatorio con distribucion uniforme
Var=0.1
#varianza=((b-a)**2)/12
#como dc=0, a=-c y b=c
c=np.sqrt(Var*3)
nn2=np.random.uniform(-c,c,N)  # Ruido de media 0 y varianza 0.1
f4, X4 = calcular_fft(nn2, fs)

# 5) Pulso rectangular
tt3 = np.arange(N) / fs

x3 = signal.square(2*np.pi*ff*tt3,duty=0.5)
f5, X5 = calcular_fft(x3, fs)

# Gráfico 1
plt.figure()
plt.plot(f1, X1)
plt.title("FFT - Señal senoidal")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid()
plt.show()

# Gráfico 2
plt.figure()
plt.plot(f2, X2)
plt.title("FFT - Señal senoidal de 2 W")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid()
plt.show()


# Gráfico 3
plt.figure()
plt.plot(f3, X3)
plt.title("FFT - Ruido Normal")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid()
plt.show()


# Gráfico 4
plt.figure()
plt.plot(f4, X4)
plt.title("FFT - Ruido Uniforme")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid()
plt.show()

# Gráfico 5
plt.figure()
plt.plot(f5, X5)
plt.title("FFT - Señal rectangular")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid()
plt.show()


# Señal de energía: tiene energía finita y potencia media 0.
# Señal de potencia: tiene energía infinita pero potencia media finita y no nula.