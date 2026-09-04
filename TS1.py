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

# Parámetros
Np = 10
fs = 20000
ff = fs/Np
N = 1000
T = 1/ff

# 1) Señal original de 2 kHz
tt1, x1 = mi_sen(vmax=1, dc=0, ff=ff, ph=0, nn=N, fs=fs)
P1=np.var(x1)

# 2) Misma señal con potencia 2W y desfasada en pi/2
Pot=2
vmax=np.sqrt(Pot*2)
tt2, x2 = mi_sen(vmax=vmax, dc=0, ff=ff, ph=np.pi/2, nn=N, fs=fs)
P2=np.var(x2)

tt_ruido = np.arange(N) / fs
# 3) Ruido aleatorio con distribucion normal
Var=0.1
nn1=np.random.normal(0,np.sqrt(Var),N)  # Ruido gaussiano de media 0 y varianza 0.1

# 4) Ruido aleatorio con distribucion uniforme
Var=0.1
nn2=np.random.uniform(0,np.sqrt(Var),N)  # Ruido gaussiano de media 0 y varianza 0.1

# 5) Pulso rectangular
tt3 = np.arange(N) / fs

x3 = signal.square(2*np.pi*ff*tt3,duty=0.5)

# Gráfico 1
plt.figure()
plt.plot(tt1, x1)
plt.title("Señal senoidal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.xlim(0, 0.01)
plt.legend()
plt.grid()
plt.show()

# Gráfico 2
plt.figure()
plt.plot(tt2, x2)
plt.title("Señal amplificada y desfasada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.xlim(0, 0.01)
plt.legend()
plt.grid()
plt.show()

# Gráfico 3
plt.figure()
plt.title("Ruido Aleatotio con distribución Normal")
plt.plot(tt_ruido,nn1, color="lime")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid()



# Gráfico 4
plt.figure()
plt.plot(tt_ruido, nn2)
plt.title("Ruido Aleatotio con distribución Uniforme")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid()
plt.show()

# Gráfico 5
plt.figure()
plt.plot(tt3, x3)
plt.title("Pulso rectangular")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.xlim(0, 0.003) 
plt.legend()
plt.grid()
plt.show()


# Señal de energía: tiene energía finita y potencia media 0.
# Señal de potencia: tiene energía infinita pero potencia media finita y no nula.