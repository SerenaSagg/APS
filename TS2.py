# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

#%%

def mi_sen(vmax, dc, ff, ph, nn, fs):    
    #vmax, dc,ff,ph tienen valor por defecto
    Ts=1/fs 
    tt=np.arange(nn)*Ts
    xx=dc+vmax*np.sin(2*np.pi*ff*tt + ph)
    return tt,xx


fs=1000 
Ts=1/fs 
N=1000
tt=np.arange(N)*Ts

tt,ss= mi_sen(vmax=np.sqrt(2), dc=0,ff=fs/N, ph=0, nn=N, fs=fs)
#%%
#Ruido
#np.random.normal(mu,sigma,size)
B=4
k=1

Vfs = 2
qq = Vfs / 2**B

#Potencia del ruido
Pq=(qq**2)/12
Pn=k*Pq #k es una escala
nn=np.random.normal(0,np.sqrt(Pn),N)  # Ruido gaussiano de media 0 y varianza Prn

sr=ss + nn  #señal+ruido
srq = np.round(sr / qq)  #Cuantización
# Volver a voltios

srq_v = srq * qq

# Gráfico comparación
plt.figure()
plt.plot(tt,sr, label="Señal con ruido", color="royalblue")
plt.plot(tt,srq_v, label="Señal cuantizada", color="fuchsia", linewidth="0.5")
plt.plot(tt,ss, label="S analógica", linestyle="--", color="aqua")
plt.title(f"Señal muestreada por un ADC de {B} bits, Vf=2V, q={qq}, k={k}")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.show()

#%%
#FFT
SR=np.fft.fft(sr)*2/N  #escalado por 1/N, y se multiplica por 2 para que se tenga en cuenta la otra mitad periodica
SRmod=np.abs(SR)
SS=np.fft.fft(ss)*2/N
SRQ=np.fft.fft(srq_v)*2/N

df=fs/N
ff=np.arange(N)*df
PotSR=(SRmod**2)
PotSR_dB=(10*np.log10(PotSR))
Pot_SRQ=(np.abs(SRQ))**2
Pot_SS=(np.abs(SS))**2
Pot_SRQ_db=(10*np.log10(Pot_SRQ))
Pot_SS_db=(10*np.log10(Pot_SS))
Pd_dB = 10 * np.log10(2 * Pq / N)
Pa_dB  = 10 * np.log10(2 * Pn / N)


bBool=ff<=fs/2
plt.figure()
plt.title(f"Señal muestreada por un ADC de {B} bits, Vf=2V, q={qq}, k={k}")
plt.plot(ff[bBool],PotSR_dB[bBool], label="Señal con ruido", color="green", linestyle=":")
plt.plot(ff[bBool],Pot_SRQ_db[bBool], label="Senal cuantizada", color="darkblue")
plt.plot(ff[bBool],Pot_SS_db[bBool], label="Señal analogica", color="orange", linestyle=":")
plt.axhline(Pa_dB, color="tomato", linestyle="--", label="Piso analógico")
plt.axhline(Pd_dB, color="violet", linestyle="--", label="Piso digital")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad de Potencia [dB]")
plt.legend()
plt.show()

#%%
# Ruido de cuantización
rq = srq_v - sr

bins=10
plt.figure()
plt.hist(rq, bins)
plt.axvline(-qq/2, color='red', linestyle='--')
plt.axvline(qq/2, color='red', linestyle='--')
plt.axhline(N/bins, color="red", linestyle="--")
plt.title(f"Ruido de cuantización con {B} bits y k={k}")
plt.ylabel("N muestras")
plt.grid(True)
plt.show()
#%%
#Ejercicio B

#Ruido
#np.random.normal(mu,sigma,size)
Bits=[4,16]
Ks= [1/10, 10]

for B in Bits:
    for k in Ks:
        Vfs = 2
        qq = Vfs / 2**B

        #Potencia del ruido
        Pq=(qq**2)/12
        Pn=k*Pq #k es una escala
        nn=np.random.normal(0,np.sqrt(Pn),N)  # Ruido gaussiano de media 0 y varianza Prn
        
        sr=ss + nn  #señal+ruido
        srq = np.round(sr / qq)  #Cuantización
        # Volver a voltios
        
        srq_v = srq * qq
        
        # Gráfico comparación
        plt.figure()
        plt.plot(tt,sr, label="Señal con ruido", color="royalblue")
        plt.plot(tt,srq_v, label="Señal cuantizada", color="fuchsia", linewidth="0.5")
        plt.plot(tt,ss, label="S analógica", linestyle="--", color="aqua")
        plt.title(f"Señal muestreada por un ADC de {B} bits, Vf=2V, q={qq}, k={k}")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud [V]")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        #%%
        #FFT
        SR=np.fft.fft(sr)*2/N  #escalado por 1/N, y se multiplica por 2 para que se tenga en cuenta la otra mitad periodica
        SRmod=np.abs(SR)
        SS=np.fft.fft(ss)*2/N
        SRQ=np.fft.fft(srq_v)*2/N
        
        df=fs/N
        ff=np.arange(N)*df
        PotSR=(SRmod**2)
        PotSR_dB=(10*np.log10(PotSR))
        Pot_SRQ=(np.abs(SRQ))**2
        Pot_SS=(np.abs(SS))**2
        Pot_SRQ_db=(10*np.log10(Pot_SRQ))
        Pot_SS_db=(10*np.log10(Pot_SS))
        Pd_dB = 10 * np.log10(2 * Pq / N)
        Pa_dB  = 10 * np.log10(2 * Pn / N)
        
        
        bBool=ff<=fs/2
        plt.figure()
        plt.title(f"Señal muestreada por un ADC de {B} bits, Vf=2V, q={qq}, k={k}")
        plt.plot(ff[bBool],PotSR_dB[bBool], label="Señal con ruido", color="green", linestyle=":")
        plt.plot(ff[bBool],Pot_SRQ_db[bBool], label="Senal cuantizada", color="darkblue")
        plt.plot(ff[bBool],Pot_SS_db[bBool], label="Señal analogica", color="orange", linestyle=":")
        plt.axhline(Pa_dB, color="tomato", linestyle="--", label="Piso analógico")
        plt.axhline(Pd_dB, color="violet", linestyle="--", label="Piso digital")
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Densidad de Potencia [dB]")
        plt.legend()
        plt.show()
        
        #%%
        # Ruido de cuantización
        rq = srq_v - sr
        
        bins=10
        plt.figure()
        plt.hist(rq, bins)
        plt.axvline(-qq/2, color='red', linestyle='--')
        plt.axvline(qq/2, color='red', linestyle='--')
        plt.axhline(N/bins, color="red", linestyle="--")
        plt.title(f"Ruido de cuantización con {B} bits y k={k}")
        plt.ylabel("N muestras")
        plt.grid(True)
        plt.show()
