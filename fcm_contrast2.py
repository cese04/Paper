from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import scipy.ndimage as ni
import time



Ima = ni.imread('meta.jpg', flatten=True)
fil, col = np.shape(Ima)
Im = np.reshape(Ima, [1,fil*col])

print(np.shape(Im))
start_time = time.time()
PE = []
fcps = []

for ncenters in range(3,8):
    #ncenters = ;
    stp = 0.1 * ncenters
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            Im, ncenters, 2, error=stp, maxiter=300, init=None)

    fcps.append(fpc)
    pe = 0
    for i in range(np.shape(u)[1]):
        for k in range(ncenters):
            pe = u[k, i] ** 2 * 2 * np.log(u[k, i]) + pe
            # pe = u[k, i] * np.log(u[k, i]) + pe

    PE.append(-(1 / (fil * col * np.log(1 / ncenters))) * pe)
    print(fcps)
    print(PE)


ncenters = np.argmax(PE)+3


cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            Im, ncenters, 2, error=stp, maxiter=300, init=None)

print("--- %s seconds ---" % (time.time() - start_time))
print(cntr)


cntr = np.reshape(cntr, ncenters)
ord = np.argsort(cntr)


u = u/np.sum(u, axis=0)
lev = 255/(ncenters-1)
imf = np.zeros_like(Ima)

for i in range(ncenters):
    imf = np.reshape(u[ord[i], :], np.shape(Ima)) * lev * i + imf

w = np.zeros(ncenters)
for i in range(ncenters):
    w[i] = np.sum(u[i])

print(w/np.sum(w))

plt.subplot(121)
plt.imshow(imf,cmap=plt.cm.gray)

plt.subplot(122)
plt.imshow(Ima,cmap=plt.cm.gray)

plt.show()
