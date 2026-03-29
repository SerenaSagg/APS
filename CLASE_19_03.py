import matplotlib.pyplot as plt
import numpy as np
#%%
def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs
    tt = np.arange(0, nn) * ts
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx
#%%
vmax=np.sqrt(2) # para que la potencia sea 1 W
dc=0
ff=1
ph=0 
nn=1000
fs=1000
snr=15  # lo definimos nosotros 
        # mide que tan fuerte es la señal comparada con el ruido 
mu = 0
tt,xx = mi_funcion_sen(vmax, dc,ff,ph,nn,fs)

potencia_senial=np.var(xx)
potencia_ruido= 10**-(snr/10)   #POTENCIA RUIDO
                    #Porque son 20db, es de poner la potencia en funcion del snr 


ruido=np.random.normal(mu,np.sqrt(potencia_ruido),nn)    # nn muestras de ruido aleatorio con media 0 
                                            # y desvio estandar (pr = 0.01), de distribucion normal 
                                            #aditivo 
yy = xx + ruido

plt.plot(tt,yy) # xx con ruido
plt.plot(tt, xx, lw=2, color = 'red') # xx sin ruido
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title(f'Frecuencia = {ff} Hz')
plt.show()
print(potencia_senial)
print(potencia_ruido)

#%%
from scipy import signal as sig 

n0 = 300
dd = np.zeros(nn)
dd[n0] = 1.

ww = sig.convolve(xx,dd) #casi el doble, suma ambos tamanios -1

#%%
kk = (1/nn) * sig.convolve(ruido, np.flip(ruido)) 

plt.plot(kk) 

plt.show()


#%% 
#CUANTIZACION

B = 3   # bits, con un incremento lineal de bits aumenta de forma lineal el dB, es decir,
        # de forma exponencial y ya ni se nota que esta cuantizado


Vfs = 3 #Volts

qq = Vfs / 2**B #paso de cuantizacion

xx_in = yy

xxq = np.round( xx_in / qq )  * qq   #codificamos a una cantidad signada de bits
                                #no se simplifica qq porque hay un redondeo.
                                #lo multiplico por qq para que me quede en la misma magnitud fisica
                                
plt.figure(2)
plt.plot(xxq, label = 'xxq') 

plt.plot(xx, label = 'xx')
plt.legend()
plt.title('Señal cuantizada')
plt.show()

#error 

ee = xxq - xx_in

#sharex mismo eje temporal 
# --- Graficación con subplots ---
fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f'Cuantización uniforme — B={B} bits, Vfs={Vfs}V, q={qq:.4f}V', fontsize=13)


# Subplot 1: señal cuantizada
ax2.plot(tt, xxq, color='tomato',    linewidth=1.2, linestyle='--', label='xxq (cuantizada)')

ax2.plot(tt, xx_in,  color='steelblue', linewidth=1,   alpha=0.4, label='xx_in (original)')
ax2.set_ylabel('Amplitud (V)')
ax2.set_title('Señal cuantizada xxq')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.5)

# Subplot 2: error de cuantización
ax3.plot(tt, ee, color='seagreen', linewidth=1)
ax3.axhline( qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'+q/2 = +{qq/2:.4f}V')
ax3.axhline(-qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'-q/2 = -{qq/2:.4f}V')
ax3.set_ylabel('Error (V)')
ax3.set_xlabel('Tiempo (s)')
ax3.set_title('Error de cuantización ee = xxq − xx')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()

#la secuencia de error tiene que ser incorrelada y distribuida uniformemente 
#no tiene estructura temporal pero en el grafico si se ve, es una senoial distorsionada porque hay estructura
#no hay estructura cuando se rompa, cuando le agrego una pequeña aleatoriedad 


# la señal que ingresa tiene un ruido que puede ser poco
# cuanto menor el snr mas se pierde el ordenamiento, si tengo una muestra la siguiente puede ser cualquier valor
# se ve altamente incorrelado y aparece en un momento concreto y tiene que ver con la potencia del ruido montado sobre la senoidal
# pasa de ser algo que se mueve como de forma "pulsal" a algo que se mueve de una forma mas rara, se rompe la correlacion
# si varia la potencia, varia la energia y aumenta el ruido. 
# el aumento de ruido va a lograr 
# que el ruido sea incorrelado es deseable, el ruido de cuantizacion tiene que ser incorrelado
# si es muy ordenado es predecible y si es predecible no va a tener un espectro plano 
# si es incorrelado la autocorrelacion va a tender a una delta (0 siempre salvo en el origen)
# si no es incorrelada esto no se cumple, va a haber varios lugares donde no es 0 
#la potencia del ruido analogico, para que roma el corrlamiento, tiene qu eser coparable con la energia del paso de cuantizacon (la mitad), tiene que ser capaz de ir para arriba y para abajo para que produzca un desnivel




#donde se provoca un salto de nivel? cuando atraviesa los niveles que veo en el grafico sin ruido (xxq con xx), siempre que tenga energia suficiente voy a provocar que vaya a un nivel y vuelva y vaya y vuelva y eso rompe la estructura de la senoidal subyacente 


#histograma -> estimador de la funcion distribucion de probabilidad 

#%%
# # ==========================================
# # ANÁLISIS ESTADÍSTICO DEL ERROR (Solo NumPy)
# # ==========================================

# # 1. Cálculo de la autocorrelación del error con NumPy
# # mode='full' nos da la correlación completa (2*nn - 1 puntos).
# # Normalizamos dividiendo por nn para estimar la potencia.
# autocorr_ee = np.correlate(ee, ee, mode='full') #/nn
# lags = np.arange(-nn + 1, nn) # Eje de retardos (lags)

# # --- Graficación del Histograma y la Autocorrelación ---
# fig3, (ax_hist, ax_corr) = plt.subplots(1, 2, figsize=(12, 5))
# fig3.suptitle('Análisis Estadístico del Error de Cuantización', fontsize=14)

# # Panel 1: Histograma del error de cuantización
# # density=True normaliza el histograma para que el área sea 1 (Densidad de Probabilidad)
# ax_hist.hist(ee, bins=10 , density=True, color='mediumorchid', alpha=0.7, edgecolor='black')
# #es importante lo de los bins 
# ax_hist.axvline( qq/2, color= 'red', linewidth=1.5, linestyle='--', label='+q/2')
# ax_hist.axvline(-qq/2, color= 'red', linewidth=1.5, linestyle='--', label='-q/2')
# #lineas verticales 

# ax_hist.set_title('Histograma (Distribución de Amplitud)')
# ax_hist.set_xlabel('Amplitud del Error (V)')
# ax_hist.set_ylabel('Densidad de Probabilidad')
# ax_hist.legend(loc='upper right')
# ax_hist.grid(True, alpha=0.3)

# # Panel 2: Autocorrelación de la secuencia de error
# ax_corr.plot(lags, autocorr_ee, color='darkorange', linewidth=1.2)
# ax_corr.set_title('Autocorrelación (Estructura Temporal)')
# ax_corr.set_xlabel('Retardo (muestras)')
# ax_corr.set_ylabel('Amplitud')
# ax_corr.grid(True, alpha=0.3)
# ax_corr.axhline(0, color='black', linewidth=0.8)

# plt.tight_layout()
# plt.show()

# autocorr_full = np.correlate (ee, ee, mode = 'full')
# lags = np.arange()


#%%
# ==========================================
# ANÁLISIS ESTADÍSTICO DEL ERROR (Simulación vs Teoría)
# ==========================================

# --- Valores Teóricos ---
varianza_teorica = (qq**2) / 12 * nn
teorica = 1 / qq

# 1. Cálculo de la autocorrelación del error con NumPy
# Descomentamos la división por 'nn' para estimar la potencia/varianza real
autocorr_ee = np.correlate(ee, ee, mode='full') #/ nn CON ESTO NORMALIZO
lags = np.arange(-nn + 1, nn) # Eje de retardos (lags)

# --- Graficación del Histograma y la Autocorrelación ---
fig3, (ax_hist, ax_corr) = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Análisis Estadístico del Error: Simulación vs Teoría', fontsize=14)

# Panel 1: Histograma del error de cuantización
ax_hist.hist(ee, bins=10, density=True, color='mediumorchid', alpha=0.7, edgecolor='black', label='Simulado')

# -- Agregamos la PDF Teórica --
# Es un rectángulo de altura 1/qq entre -q/2 y q/2
eje_e = np.linspace(-qq, qq, 1000)
pdf_uniforme = np.where((eje_e >= -qq/2) & (eje_e <= qq/2), teorica, 0)
ax_hist.plot(eje_e, pdf_uniforme, color='blue', linewidth=2.5, label='Teórica (Uniforme)')

ax_hist.axvline( qq/2, color='red', linewidth=1.5, linestyle='--', label='+q/2')
ax_hist.axvline(-qq/2, color='red', linewidth=1.5, linestyle='--', label='-q/2')

ax_hist.set_title('Histograma vs Teórica')
ax_hist.set_xlabel('Amplitud del Error (V)')
ax_hist.set_ylabel('Densidad de Probabilidad')
ax_hist.legend(loc='upper right', fontsize=9)
ax_hist.grid(True, alpha=0.3)


# Panel 2: Autocorrelación de la secuencia de error
ax_corr.plot(lags, autocorr_ee, color='darkorange', linewidth=1.2, label='Simulada')

# -- Agregamos la Autocorrelación Teórica --
# Es un impulso en 0 con amplitud q^2/12 (y 0 en el resto)
autocorr_teorica = np.zeros_like(lags, dtype=float)
autocorr_teorica[lags == 0] = varianza_teorica

# Dibujamos la teórica con una línea punteada y un punto marcado en el centro
ax_corr.plot(lags, autocorr_teorica, color='blue', linestyle='--', linewidth=1.5, label=f'Teórica (q²/12 = {varianza_teorica:.4f})')
ax_corr.plot(0, varianza_teorica, marker='o', color='blue')

ax_corr.set_title('Autocorrelación vs Varianza Teórica')
ax_corr.set_xlabel('Retardo (muestras)')
ax_corr.set_ylabel('Potencia')
ax_corr.legend(loc='upper right', fontsize=9)
ax_corr.grid(True, alpha=0.3)
ax_corr.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()

#histograma es una aproximacion a la distribucion teorica que quiero tener 
# si veo que en algun lado se pasa de la teorica, es una hipotesis fuerte para decir qu eno se corresponde tanto (aunque igual esta bastant bien, le voy a poner 0 de ruido para que no lo sea )

## DEMUESTRO SIN RUIDO NO ES UNIFORME
#%%
# ==========================================
# ANÁLISIS ESTADÍSTICO DEL ERROR (Simulación vs Teoría)
# ==========================================
xxq_sin_ruido = np.round(xx / qq) * qq
ee = xxq_sin_ruido - xx

# --- Valores Teóricos ---
varianza_teorica = (qq**2) / 12 * nn
teorica = 1 / qq

# 1. Cálculo de la autocorrelación del error con NumPy
# Descomentamos la división por 'nn' para estimar la potencia/varianza real
autocorr_ee = np.correlate(ee, ee, mode='full') #/nn CON ESTO NORMALIZO
lags = np.arange(-nn + 1, nn) # Eje de retardos (lags)

# --- Graficación del Histograma y la Autocorrelación ---
fig3, (ax_hist, ax_corr) = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Análisis Estadístico del Error: Simulación vs Teoría', fontsize=14)

# Panel 1: Histograma del error de cuantización
ax_hist.hist(ee, bins=10, density=True, color='mediumorchid', alpha=0.7, edgecolor='black', label='Simulado')

# -- Agregamos la PDF Teórica --
# Es un rectángulo de altura 1/qq entre -q/2 y q/2
eje_e = np.linspace(-qq, qq, 1000)
pdf_uniforme = np.where((eje_e >= -qq/2) & (eje_e <= qq/2), teorica, 0)
ax_hist.plot(eje_e, pdf_uniforme, color='blue', linewidth=2.5, label='Teórica (Uniforme)')

ax_hist.axvline( qq/2, color='red', linewidth=1.5, linestyle='--', label='+q/2')
ax_hist.axvline(-qq/2, color='red', linewidth=1.5, linestyle='--', label='-q/2')

ax_hist.set_title('Histograma vs Teórica')
ax_hist.set_xlabel('Amplitud del Error (V)')
ax_hist.set_ylabel('Densidad de Probabilidad')
ax_hist.legend(loc='upper right', fontsize=9)
ax_hist.grid(True, alpha=0.3)


# Panel 2: Autocorrelación de la secuencia de error
ax_corr.plot(lags, autocorr_ee, color='darkorange', linewidth=1.2, label='Simulada')


# -- Agregamos la Autocorrelación Teórica (Estilo Delta Discreta) --
autocorr_teorica = np.zeros_like(lags, dtype=float)
autocorr_teorica[lags == 0] = varianza_teorica
# Usamos STEM para dibujar el "palo" en el retardo 0
ax_corr.stem([0], [varianza_teorica], linefmt='b--', markerfmt='bo', basefmt=" ", label=f'Teórica (q²/12 = {varianza_teorica:.4f})')

# Usamos stem en el retardo 0 para dibujar el "bastón" vertical característico de la delta
ax_corr.stem([0], [varianza_teorica], linefmt='b--', markerfmt='bo', basefmt=" ", label=f'Teórica (q²/12 = {varianza_teorica:.4f})')

# Dibujamos la teórica con una línea punteada y un punto marcado en el centro
ax_corr.plot(lags, autocorr_teorica, color='blue', linestyle='--', linewidth=1.5, label=f'Teórica (q²/12 = {varianza_teorica:.4f})')
ax_corr.plot(0, varianza_teorica, marker='o', color='blue')

ax_corr.set_title('Autocorrelación vs Varianza Teórica')
ax_corr.set_xlabel('Retardo (muestras)')
ax_corr.set_ylabel('Potencia')
ax_corr.legend(loc='upper right', fontsize=9)
ax_corr.grid(True, alpha=0.3)
ax_corr.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()

#siempre 10 o 20 bins porque sino puede que no caigan muestras ahi 

#%% FFT
#

from numpy import fft 

#devuelve la transformada de fourier discreta 

#le paso una secuencia y me lo transforma
#si le pasamos una senoidal nos devuelve dos deltas 
# a partir de una secuencia real, nos devuelve una secuencia de complejos
# vamos a ver 

XX = np.fft.fft(xx) # si le pasamos 1000 numeros reales me devuelve 1000 numeros complejos

XXmod = np.abs(XX) # MODULO
XXphase = np.angle(XX) # FASE


plt.figure()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle('DFT de xx', fontsize=13)

ax1.plot(XXmod, color ='tomato', linewidth = 1, label ='Modulo')
ax1.legend(fontsize=8)
ax2.plot(XXphase, color='mediumpurple', linewidth = 1, label='Fase')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()


#graficos modulo fase, distribucion 


#decir por que se ve asi

#sin ruido, con 30 db snr, 20 snt
#hacer con 15db en snr 
#potencia de 15db y el resultado de autocorrelacion, histograma, etc














