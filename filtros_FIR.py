#%% CARGA ECG
from scipy import signal as sig
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

mat_struct = sio.loadmat("./ECG_TP4.mat")
ecg = mat_struct["ecg_lead"].flatten()

fs= 1000

#%% DISEÑO FIR PASABANDA ECG

cant_coef = 2001

ws1 = 0.1
wp1 = 1
wp2 = 35
ws2 = 45

ripple = 1
attenuation = 40

nyq = fs/ 2

frecs = np.array([0, ws1, wp1, wp2, ws2, nyq]) / nyq
gains_db = np.array([-attenuation, -attenuation, -ripple, -ripple, -attenuation, -attenuation])
gains = 10**(gains_db/20)

coef_hm = sig.firwin2(cant_coef, frecs, gains, window="hamming")
N = len(coef_hm)
demora=int((N-1)/2)
#%% FILTRADO

ecg_fir = sig.lfilter(coef_hm, [1], ecg)

#%% GRAFICAR ECG ORIGINAL VS FILTRADO

plt.figure(figsize=(12,6), tight_layout=True)

plt.plot(ecg, label="Original", alpha=0.7)
plt.plot(ecg_fir, label="FIR filtrado")

plt.title("ECG original vs FIR filtrado")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()


#%%
#%% FILTRO FIR CON CUADRADOS MINIMOS 
ws1c = 0.126     # Banda de stop baja
wp1c = 0.5      # Inicio banda de paso
wp2c = 35       # Fin banda de paso
ws2c = 37       # Banda de stop alta


 
weight = np.array([2,1,1])
b_cuadradosmin = sig.firls (cant_coef , bands =np.array([0., ws1c, wp1c, wp2c, ws2c, fs/2]), desired=[0, 0, 1, 1, 0, 0], weight = weight, fs = fs ) 
# Respuesta en frecuencia del FIR
wc, hc = sig.freqz(b_cuadradosmin, worN=ww, fs=fs)

modulo_dbc = 20 * np.log10(np.abs(hc) + 1e-12)


#%% Grafico con plantilla de diseño sombreada

fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)

ax1.set_title("Frequency Response of FIR Filter - Cuadrados mínimos")

ax1.plot(wc, modulo_dbc, 'b', linewidth=1.8, label='Filtro diseñado')

piso_grafico = -125
techo_grafico = 5

# Banda de parada baja
ax1.fill_between(
    [0, ws1],
    -gstop,
    techo_grafico,
    color='plum',
    alpha=0.15,
    label='Plantilla'
)

ax1.plot(
    [0, ws1],
    [-gstop, -gstop],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Banda de paso
ax1.fill_between(
    [wp1, wp2],
    piso_grafico,
    -gpass,
    color='violet',
    alpha=0.15
)

ax1.plot(
    [wp1, wp2],
    [-gpass, -gpass],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Zona superior banda de paso
ax1.fill_between(
    [wp1, wp2],
    3,
    techo_grafico,
    color='plum',
    alpha=0.15
)

ax1.plot(
    [wp1, wp2],
    [3, 3],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Banda de parada alta
ax1.fill_between(
    [ws2, fs/2],
    -gstop,
    techo_grafico,
    color='plum',
    alpha=0.15
)

ax1.plot(
    [ws2, fs/2],
    [-gstop, -gstop],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Lineas verticales
ax1.axvline(ws1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp2, color='k', linestyle=':', alpha=0.5)
ax1.axvline(ws2, color='k', linestyle=':', alpha=0.5)

ax1.set_ylabel("Amplitude in dB", color='b')
ax1.set_xlabel("Frequency [Hz]")
ax1.set_xlim(0, fs/2)
ax1.set_ylim([piso_grafico, techo_grafico])
ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')

plt.show()


###################################
#%% Regiones de interés sin ruido #
###################################
cant_muestras=len(ecg)
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_fir[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
###################################
#%% Regiones de interés con ruido #
###################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_fir[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
    
    
#%%  FILTRO FIR CON Parks-Mc Clellan-Remez
ws1r = 0.126     # Banda de stop baja
wp1r = 0.5      # Inicio banda de paso
wp2r = 35       # Fin banda de paso
ws2r = 36       # Banda de stop alta

numtapsr = 2001

weightr = np.array([2,1,1])
b_remez = sig.remez (numtapsr , bands =np.array([0., ws1r, wp1r, wp2r, ws2r, fs/2]), desired=[0, 1, 0], weight = weightr, type='hilbert', maxiter=25, grid_density=16, fs = fs ) 
# Respuesta en frecuencia del FIR
wr, hr = sig.freqz(b_remez, worN=ww, fs=fs)

modulo_dbr = 20 * np.log10(np.abs(hr) + 1e-12)


#%% Grafico con plantilla de diseño sombreada

fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)

ax1.set_title("Frequency Response of FIR Filter - Parks-Mc Clellan-Remez")

ax1.plot(wr, modulo_dbr, 'b', linewidth=1.8, label='Filtro diseñado')

piso_grafico = -125
techo_grafico = 5

# Banda de parada baja
ax1.fill_between(
    [0, ws1],
    -gstop,
    techo_grafico,
    color='plum',
    alpha=0.15,
    label='Plantilla'
)

ax1.plot(
    [0, ws1],
    [-gstop, -gstop],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Banda de paso
ax1.fill_between(
    [wp1, wp2],
    piso_grafico,
    -gpass,
    color='violet',
    alpha=0.15
)

ax1.plot(
    [wp1, wp2],
    [-gpass, -gpass],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Zona superior banda de paso
ax1.fill_between(
    [wp1, wp2],
    3,
    techo_grafico,
    color='plum',
    alpha=0.15
)

ax1.plot(
    [wp1, wp2],
    [3, 3],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Banda de parada alta
ax1.fill_between(
    [ws2, fs/2],
    -gstop,
    techo_grafico,
    color='plum',
    alpha=0.15
)

ax1.plot(
    [ws2, fs/2],
    [-gstop, -gstop],
    'k--',
    linewidth=1,
    alpha=0.7
)

# Lineas verticales
ax1.axvline(ws1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp2, color='k', linestyle=':', alpha=0.5)
ax1.axvline(ws2, color='k', linestyle=':', alpha=0.5)

ax1.set_ylabel("Amplitude in dB", color='b')
ax1.set_xlabel("Frequency [Hz]")
ax1.set_xlim(0, fs/2)
ax1.set_ylim([piso_grafico, techo_grafico])
ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')

plt.show()
