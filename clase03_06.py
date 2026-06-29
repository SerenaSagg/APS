import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal as sig

fs_ecg = 1000  # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
# sio.whosmat('ECG_TP4.mat')

mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()

# N = len(ecg_one_lead)

## Procesar el ECG

yy = sig.sosfilt(sos, ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead)
plt.plot(yy)