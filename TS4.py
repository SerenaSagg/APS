import matplotlib.pyplot as plt
import numpy as np

def mi_sen(vmax, dc, omega1, ph, nn, fs):    
    #vmax, dc,ff,ph tienen valor por defecto
    Ts=1/fs 
    tt=np.arange(nn)*Ts
    xx=dc+vmax*np.sin(2*np.pi*omega1*np.arange(nn) + ph)
    return tt,xx
fs=1000
N=1000
r=200
SNR=10
Pr=10**(-SNR/10)

omega0=np.pi/2
fr=np.random.uniform(-2,2,r).reshape((r,1))
omega1= omega0 + fr*(2*np.pi)/N
na=np.random.normal(0,np.sqrt(Pr), (r,N))


#omega1=omega1.reshape(r,1)
arrayR=np.array([r,0])
arrayN=np.array([0,N])

vecR=np.tile(fr,(1,r))
vecN=np.tile(fr, (1,N))

tt,xx= mi_sen(vmax=2, dc=0,omega1=omega1, ph=0, nn=N, fs=fs) 
XX=xx+na


plt.title("Matriz RxN ")
plt.plot(tt,XX[0])
plt.grid(True)
plt.show()