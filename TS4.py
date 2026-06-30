import numpy as np
from scipy.signal import windows


def mi_sen_omega(vmax, dc, Omega, ph, nn):
    n = np.arange(nn)
    xx = dc + vmax*np.sin(Omega*n + ph)
    return n, xx

# Parámetros
N = 1000
M = 200

Omega0 = np.pi/2
a0 = np.sqrt(2)   # potencia senoidal = 1 W

SNRs = [3, 10]

# Ventanas

ventanas = {
    "Rectangular": np.ones(N),
    "Flat-top": windows.flattop(N),
    "Blackman-Harris": windows.blackmanharris(N),
    "Hann": windows.hann(N)
}


Omega = 2*np.pi*np.arange(N)/N
k0 = int(Omega0/(2*np.pi/N))

for SNR in SNRs:

    Ps = 1
    Pn = Ps/(10**(SNR/10))
    sigma = np.sqrt(Pn)

    print("\n====================================")
    print("SNR =", SNR, "dB")
    print("sigma =", sigma)
    print("====================================")

    for nombre, w in ventanas.items():

        a_est = np.zeros(M)
        Omega_est = np.zeros(M)
        Omega_real = np.zeros(M)


        for j in range(M):

            fr = np.random.uniform(-2, 2)
            Omega1 = Omega0 + fr*(2*np.pi/N)

            Omega_real[j] = Omega1

            n, x = mi_sen_omega(
                vmax=a0,
                dc=0,
                Omega=Omega1,
                ph=0,
                nn=N
            )

            ruido = np.random.normal(0, sigma, N)

            x = x + ruido

            xw = x*w

            Xw = np.fft.fft(xw)

            # Estimador de amplitud
            a_est[j] = (2/(N))*np.abs(Xw[k0])

            # Estimador de frecuencia
            kmax = np.argmax(np.abs(Xw[:N//2]))
            Omega_est[j] = Omega[kmax]


# Sesgo y varianza amplitud

        mu_a = np.mean(a_est)
        sesgo_a = mu_a - a0
        var_a = np.var(a_est)

# Sesgo y varianza frecuencia


        error_Omega = Omega_est - Omega_real

        sesgo_Omega = np.mean(error_Omega)
        var_Omega = np.var(error_Omega)

        print("\n------------------------------------")
        print("Ventana:", nombre)
        print("------------------------------------")

        print("Estimación de amplitud")
        print("Media:", mu_a)
        print("Sesgo:", sesgo_a)
        print("Varianza:", var_a)

        print("\nEstimación de frecuencia")
        print("Sesgo:", sesgo_Omega)
        print("Varianza:", var_Omega)