# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:23:13 2025

@author: sere2
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

def energia(x):
    E = np.sum(np.abs(x)**2)
    return E

def potencia(x):
    N = len(x)
    return (1/N) * np.sum(np.abs(x)**2)

def productoInt(x, y, Ts):
    return np.sum(x * y) * Ts

# 1) Señal sinusoidal
print("\n1) Señal sinusoidal")
A=1
f1=2000
fs=10*f1
t=np.arange(0,0.01,1/fs)
phi=0

s1 = A*np.sin(2*np.pi*f1*t+phi)

plt.plot(t,s1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.title("s1: Señal Sinusoidal")
plt.show()

print("Ts=", 1/fs, "s")
print("N=", len(s1))
P=potencia(s1)
print("Potencia:", round(P,2),"W" )

#Autocorrelacion
Rs1s1=np.correlate(s1,s1, mode="full")
retardo = np.arange(-(len(s1)-1), len(s1))

plt.plot(retardo,Rs1s1, color="orange")
plt.title("Autocorrelacion")
plt.grid()
plt.show()


# 2) Señal amplificada y desfasada
print("\n2) Señal amplificada y desfasada")
A=3
f2=2000
fs=10*f2
t=np.arange(0,0.01,1/fs)
phi=np.pi/2

s2 = A*np.sin(2*np.pi*f2*t+phi)

plt.plot(t,s2, color="green")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.title("s2: Señal Amplificada y Desfasada")
plt.show()

print("Ts=", 1/fs, "s")
print("N=", len(s2))
P=potencia(s2)
print("Potencia:", round(P,2),"W" )

# Ortogonalidad con s1
ref = np.sin(2*np.pi*f1*t+0)
O=productoInt(ref,s2,1/fs)
if(np.isclose(O,0)):
    print("Señales ortogonales")
else:
    print("Señales no ortogonales")  
    
#Correlacion
Rs1s2=np.correlate(s1,s2, mode="full")
retardo = np.arange(-(len(s1)-1), len(s2))

plt.plot(retardo,Rs1s2, color="orange")
plt.grid()
plt.title("Correlacion")
plt.show()


# 3) Señal modulada 
print("\n3) Señal modulada")
A=1
f3=2000
phi=0
B=1
f4=f3/2
Ts=1/(10*(f3+f4))
t=np.arange(0,0.01,Ts)

s3a = A*np.sin(2*np.pi*f3*t+phi)   # portadora
s3b = B*np.sin(2*np.pi*f4*t+phi)   # moduladora
s3  = s3a * s3b

plt.subplot(3,1,1)
plt.plot(t,s3a, label="s1 original")
plt.legend(); plt.grid()

plt.subplot(3,1,2)
plt.plot(t,s3b, label="s3 moduladora", color="red")
plt.legend(); plt.grid()

plt.subplot(3,1,3)
plt.plot(t,s3, label="s3 modulada", color="orange")
plt.legend(); plt.grid()
plt.show()

print("Ts=", Ts, "s")
print("N=", len(s3))
P=potencia(s3)
print("Potencia:", round(P,2),"W" )

ref = np.sin(2*np.pi*f3*t+phi)
O=productoInt(ref,s3,Ts)
if(np.isclose(O,0)):
    print("Señales ortogonales")
else:
    print("Señales no ortogonales")   


#Correlacion
Rs1s3=np.correlate(s1,s3, mode="full")
retardo = np.arange(-(len(s1)-1), len(s3))

plt.plot(retardo,Rs1s3, color="orange")
plt.grid()
plt.title("Correlacion")
plt.show()

# 4) Señal recortada
print("\n4) Señal recortada")
A=1
f5=2000
fs=10*f5
t=np.arange(0,0.001,1/fs)
phi=0

s4a = A*np.sin(2*np.pi*f5*t+phi)
Amax = 0.75*A
s4  = np.clip(s4a, -Amax, Amax)

plt.axhline(y=Amax, color="red", linestyle="--", label="A max")
plt.axhline(y=-Amax, color="red", linestyle="--")
plt.plot(t,s4)
plt.legend()
plt.title("s4: Señal Recortada al 75%")
plt.grid()
plt.show()

print("Ts=", 1/fs,"s")
print("N=", len(s4))
P=potencia(s4)
print("Potencia:", round(P,2),"W" )

ref = np.sin(2*np.pi*f5*t+phi)
O=productoInt(ref,s4,1/fs)
if(np.isclose(O,0)):
    print("Señales ortogonales")
else:
    print("Señales no ortogonales")  
    
#Correlacion
Rs1s4=np.correlate(s1,s4, mode="full")
retardo = np.arange(-(len(s1)-1), len(s4))

plt.plot(retardo,Rs1s4, color="orange")
plt.grid()
plt.title("Correlacion")
plt.show()

# 5) Señal cuadrada
print("\n5) Señal cuadrada")
f6 = 4000         
fs = 100*f6
t = np.arange(0,0.001,1/fs)

s5 = square(2*np.pi*f6*t)

plt.plot(t, s5)
plt.title("s5: Señal Cuadrada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

print("Ts=", 1/fs,"s")
print("N=", len(s5))
P=potencia(s5)
print("Potencia:", round(P,2),"W" )

ref = np.sin(2*np.pi*2000*t+0)   
O=productoInt(ref,s5,1/fs)
if(np.isclose(O,0)):
    print("Señales ortogonales")
else:
    print("Señales no ortogonales")   

#Correlacion
Rs1s5=np.correlate(s1,s5, mode="full")
retardo = np.arange(-(len(s1)-1), len(s5))

plt.plot(retardo,Rs1s5, color="orange")
plt.title("Correlacion")
plt.grid()
plt.show()

# 6) Pulso rectangular
print("\n6) Pulso rectangular")
N = 1000       
fs = 10000        # Hz (suficiente para una senoide de 2 kHz)       # número de muestras
t  = np.arange(N) / fs  
s6 = np.zeros(N)
for i in range(N):
    if (i >= 10) and (i <= 20):
        s6[i]=1


plt.plot(t, s6)
plt.xlabel("Tiempo [ms]")
plt.ylabel("Amplitud")
plt.title("s6: Pulso Rectangular")
plt.grid()
plt.show()

print("Ts= 1 ms")
print("N=", N)
E=energia(s6)
print("Energia:", round(E,2),"J")

ref = np.sin(2*np.pi*2000*t+0)   
O=productoInt(ref,s6,1/fs)
if(np.isclose(O,0)):
    print("Señales ortogonales")
else:
    print("Señales no ortogonales")  
    
#Correlacion
Rs1s6=np.correlate(s1,s6, mode="full")
retardo = np.arange(-(len(s1)-1), len(s6))

plt.plot(retardo,Rs1s6, color="orange")
plt.title("Correlacion con S1")
plt.grid()
plt.show()




#Propiedad Trigonometrica

f=2000
fs=10*f
t=np.arange(0,0.01,1/fs )

s1=np.sin(2*np.pi*f*t)  #alfa
s2=np.sin(np.pi*f*t)  #beta
s3=np.cos(2*np.pi*f*t - np.pi*f*t)
s4=np.cos(2*np.pi*f*t + np.pi*f*t)

plt.subplot(2,1,1)
plt.plot(t,2*s1*s2, color="red", label="2sen(α)sen(β)")
plt.title("Propiedad Trigonométrica")
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t,(s3-s4),color="green", label="cos(α-β)-cos(α+β)")
plt.legend()
plt.grid()
plt.show()

