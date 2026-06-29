#TS7
from scipy import signal
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#%% CARGA DEL ARCHIVO

mat_struct = sio.loadmat("./ECG_TP4.mat")
print(mat_struct.keys())   # para ver los nombres de las variables

ecg = mat_struct["ecg_lead"].flatten()

#fs = 1 kHz
fs = 1000
nyq_frec = fs / 2

# Vector de tiempo
t = np.arange(len(ecg)) / fs

#%% PLANTILLA PASABANDA ECG

wp1 = 0.5     # inicio banda pasante [Hz]
ws1 = 0.1    # rechazo baja frecuencia [Hz]
wp2 = 35     # fin banda pasante [Hz]
ws2 = 45     # rechazo alta frecuencia [Hz]

gpass = 1    # pérdida máxima en banda pasante [dB]
gstop = 40   # atenuación mínima en rechazo [dB]

wp = [wp1, wp2]
ws = [ws1, ws2]

#ftype = "butter"
#ftype = "cheby1"
#ftype = "cheby2"
ftype = "cauer"

#%% DISEÑO DEL FILTRO EN SOS

sos_coef = signal.iirdesign(
    wp, ws,
    gpass/2, gstop/2,
    analog=False,
    ftype=ftype,
    output="sos",
    fs=fs
)
sos_butter = signal.iirdesign(
    wp, ws,
    gpass/2, gstop/2,
    analog=False,
    ftype='butter',
    output="sos",
    fs=fs
)
sos_cauer = signal.iirdesign(
    wp, ws,
    gpass/2, gstop/2,
    analog=False,
    ftype='cauer',
    output="sos",
    fs=fs
)
sos_cheby1 = signal.iirdesign(
    wp, ws,
    gpass/2, gstop/2,
    analog=False,
    ftype='cheby1',
    output="sos",
    fs=fs
)
sos_cheby2 = signal.iirdesign(
    wp, ws,
    gpass/2, gstop/2,
    analog=False,
    ftype='cheby2',
    output="sos",
    fs=fs
)

orden = sos_coef.shape[0] * 2

print("Tipo de filtro:", ftype)
print("Cantidad de secciones SOS:", sos_coef.shape[0])
print("Orden aproximado:", orden)

#%% PLANTILLA DE DISEÑO

frecs = np.array([0, ws1, wp1, wp2, ws2, nyq_frec])
gains_db = np.array([-gstop, -gstop, -gpass, -gpass, -gstop, -gstop])

plt.figure(tight_layout=True)
plt.plot(frecs, gains_db, "o-")
plt.title("Plantilla de diseño pasabanda para ECG")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Ganancia [dB]")
plt.grid(True)
plt.show()

#%%
#PLANTILLA + RESPUESTA DEL FILTRO PASABANDA ECG
ww=np.concat([np.logspace(start=-2, stop=0, num=200), 
             np.linspace(start=2, stop=35, num=50),
             np.logspace(start=1.55, stop=1.65, num=100), 
             np.linspace(start=46, stop=fs/2, num=50)]) #construye un vector de frecuencias ww con muchos puntos donde te interesa mirar el filtro y pocos donde no

# Respuesta en frecuencia con SOS
w, h = signal.sosfreqz(sos_coef, worN=ww, fs=fs)

fig, ax1 = plt.subplots(figsize=(12, 5), tight_layout=True)
ax1.set_title("Plantilla de Diseño - Filtro Pasa Banda ECG")

# Curva del filtro
ax1.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)),
         'b', linewidth=1.8, label='Filtro diseñado')

piso_grafico = -125
techo_grafico = 10

# Rechazo baja frecuencia: 0 a ws1
ax1.fill_between([0, ws1], -gstop, techo_grafico,
                 color='green', alpha=0.15, label='Zonas prohibidas')
ax1.plot([0, ws1], [-gstop, -gstop], 'k--', linewidth=1)

# Banda pasante: wp1 a wp2
# Prohibido estar por debajo de -gpass
ax1.fill_between([wp1, wp2], piso_grafico, -gpass,
                 color='green', alpha=0.15)
ax1.plot([wp1, wp2], [-gpass, -gpass], 'k--', linewidth=1)

# Rechazo alta frecuencia: ws2 a Nyquist
ax1.fill_between([ws2, fs/2], -gstop, techo_grafico,
                 color='green', alpha=0.15)
ax1.plot([ws2, fs/2], [-gstop, -gstop], 'k--', linewidth=1)

# Líneas verticales de las frecuencias de plantilla
ax1.axvline(ws1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp1, color='k', linestyle=':', alpha=0.5)
ax1.axvline(wp2, color='k', linestyle=':', alpha=0.5)
ax1.axvline(ws2, color='k', linestyle=':', alpha=0.5)

ax1.set_ylabel('Amplitud [dB]')
ax1.set_xlabel('Frecuencia [Hz]')
ax1.set_xlim(0, 60)       # para ver bien la zona útil del ECG
ax1.set_ylim(piso_grafico, techo_grafico)

ax1.grid(True, which='both', linestyle='-', alpha=0.4)
ax1.legend(loc='lower right')

plt.show()

#%% RESPUESTA EN FRECUENCIA

f, H = signal.sosfreqz(sos_coef, worN=ww, fs=fs)

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)

axs[0].plot(f, 20*np.log10(np.maximum(np.abs(H), 1e-12)))
axs[0].set_title(f"Respuesta en frecuencia - IIR {ftype} en SOS")
axs[0].set_ylabel("Módulo [dB]")
axs[0].grid(True)

phase = np.unwrap(np.angle(H))

axs[1].plot(f, phase)
axs[1].set_ylabel("Fase [rad]")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].grid(True)

plt.show()

#%% DIAGRAMA DE POLOS Y CEROS

z, p, k = signal.sos2zpk(sos_coef)

plt.figure(tight_layout=True)

theta = np.linspace(0, 2*np.pi, 500)
plt.plot(np.cos(theta), np.sin(theta), "--", label="Circunferencia unidad")

plt.scatter(np.real(z), np.imag(z), marker="o", label="Ceros")
plt.scatter(np.real(p), np.imag(p), marker="x", label="Polos")

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

plt.title("Diagrama de polos y ceros")
plt.xlabel("Parte real")
plt.ylabel("Parte imaginaria")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

#%% RETARDO DE GRUPO

Omega = 2*np.pi*f/fs
gd= -np.diff(phase)/np.diff(Omega)
gd= np.append(gd[0], gd)

plt.figure(tight_layout=True)
plt.plot(ww, gd)
plt.title("Retardo de grupo")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Retardo de grupo [muestras]")
plt.grid(True)
plt.show()


#%% APLICACIÓN DEL FILTRO AL ECG

ecg_filtrado = signal.sosfiltfilt(sos_coef, ecg)
ecgf_butt = signal.sosfiltfilt(sos_butter, ecg)
ecgf_cauer = signal.sosfiltfilt(sos_cauer, ecg)
ecgf_cheby1 = signal.sosfiltfilt(sos_cheby1, ecg)
ecgf_cheby2 = signal.sosfiltfilt(sos_cheby2, ecg)

#%% COMPARACIÓN ECG ORIGINAL VS FILTRADO

plt.figure(figsize=(12, 6), tight_layout=True)


plt.plot(ecg, label='Original')
plt.plot(ecg_filtrado, color='green', label='Filtrado')
plt.title("ECG original vs Filtrado")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

#%%
###################################
#Regiones de interés sin ruido #
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
    plt.plot(zoom_region, ecgf_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecgf_cauer[zoom_region], label='Cauer')
    plt.plot(zoom_region, ecgf_cheby1[zoom_region], label='Cheby1')
    plt.plot(zoom_region, ecgf_cheby2[zoom_region], label='Cheby2')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
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
    plt.plot(zoom_region, ecgf_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ecgf_cauer[zoom_region], label='Cauer')
    plt.plot(zoom_region, ecgf_cheby1[zoom_region], label='Cheby1')
    plt.plot(zoom_region, ecgf_cheby2[zoom_region], label='Cheby2')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

