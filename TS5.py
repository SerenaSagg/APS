import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat, wavfile



ruta_ecg = r"C:\Users\sere2\pdstestbench\ECG_TP4.mat"
ruta_ppg = r"C:\Users\sere2\pdstestbench\PPG.csv"
ruta_cuca = r"C:\Users\sere2\pdstestbench\la cucaracha.wav"
ruta_silb = r"C:\Users\sere2\pdstestbench\silbido.wav"

#=================================================
# FUNCIÓN PSD + ANCHO DE BANDA (95% potencia)
#=================================================

def calcular_psd_bw(x, fs, nombre):

    x = x - np.mean(x)

    # PSD mediante Welch
    f, PSD = signal.welch(
        x,
        fs=fs,
        window='hann',
        nperseg=1000
    )

    # Potencia acumulada
    P_acum = np.cumsum(PSD)

    # Potencia total
    P_total = P_acum[-1]

    # Normalización
    P_norm = P_acum / P_total

    # Frecuencia que contiene el 95% de la potencia
    indice = np.where(P_norm >= 0.95)[0][0]

    BW = f[indice]

    #-----------------------------
    # Gráfico
    #-----------------------------

    plt.figure(figsize=(8,4))

    plt.semilogy(f, PSD)

    plt.axvline(BW,
                color='red',
                linestyle='--',
                label=f'BW = {BW:.2f} Hz')

    plt.title(f'PSD - {nombre}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [V²/Hz]')
    plt.grid(True)
    plt.legend()

    plt.show()

    return BW

# ECG

mat = loadmat(ruta_ecg)

ecg = mat["ecg_lead"].squeeze()

fs_ecg = 1000

BW_ecg = calcular_psd_bw(ecg, fs_ecg, "ECG")


# PPG

ppg = pd.read_csv(ruta_ppg)

senal_ppg = ppg.iloc[:,0].values

fs_ppg = 400

BW_ppg = calcular_psd_bw(senal_ppg,
                         fs_ppg,
                         "PPG")


# AUDIO

fs_audio, audio = wavfile.read(ruta_cuca)

if audio.ndim > 1:
    audio = audio[:,0]

BW_audio = calcular_psd_bw(audio,
                           fs_audio,
                           "La Cucaracha")

# SILBIDO


fs_silb, silb = wavfile.read(ruta_silb)

if silb.ndim > 1:
    silb = silb[:,0]

BW_silb = calcular_psd_bw(silb,
                          fs_silb,
                          "Silbido")



tabla = pd.DataFrame({

    "Señal":[
        "ECG",
        "PPG",
        "La Cucaracha",
        "Silbido"
    ],

    "Ancho de banda (95% potencia) [Hz]":[
        BW_ecg,
        BW_ppg,
        BW_audio,
        BW_silb
    ]

})

print("\n")
print(tabla)