# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 15:20:12 2017

@author: CarlosEmiliano
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import scipy.ndimage as ni
import time

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = np.reshape(image_equalized, image.shape)
    return image_equalized.reshape(image.shape), cdf


eps = 0.0001

Ima = ni.imread('knee2.jpg', flatten=True)

image_histogram, bins = np.histogram(Ima.flatten(), 256, normed=True)


fil, col = np.shape(Ima)

Im = np.reshape(Ima, [1, fil * col])

print(np.shape(Im))
start_time = time.time()


image_histogram, bins = np.histogram(Ima.flatten(), 256, normed=True)

PE = []
fcps = []
cnt = []
for ncenters in range(3,7):
    #ncenters = ;
    stp = 0.2 * ncenters
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            Im, ncenters, 2, error=stp, maxiter=300, init=None)
    
    cnt.append(cntr)
    fcps.append(fpc)
    pe = 0
    for i in range(np.shape(u)[1]):
        for k in range(ncenters):
            pe = u[k, i] ** 2 * np.log(u[k, i]**2) + pe
            #pe = u[k, i] ** 2 * 2 * np.log(u[k, i]) + pe
            # pe = u[k, i] * np.log(u[k, i]) + pe

    PE.append(-(1 / (fil * col * np.log(1 / ncenters))) * pe)
    print(fcps)
    print(PE)
    
ncenters = np.argmax(PE) + 3
cntr = cnt[np.argmax(PE)]
print ncenters
'''stp = 0.2 * ncenters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            Im, ncenters, 2, error=stp, maxiter=300, init=None)'''

print("--- %s seconds ---" % (time.time() - start_time))


cntr = np.reshape(cntr, ncenters)
ord = np.argsort(cntr)


lev = (255) / (ncenters - 1)

w = np.zeros(ncenters)
for i in range(ncenters):
    w[i] = np.sum(u[i])

w = np.sqrt(w / np.sum(w) * 255)*6
print(w)

#x = np.arange(0, 256, 1)

me = []
mx = np.max(Im)
mx = mx.astype(int)
print mx
fcm_gr = ctrl.Antecedent(np.arange(-1, np.max(Im) + 2), 'grupos')
fcm_sl = ctrl.Consequent(np.arange(0,256,1), 'salida')
for i in range(ncenters):
    
    if i == 0:
        abc = [0, 0, cntr[ord[i+1]]]
        #print(abc)
    elif i == ncenters-1:
        abc = [cntr[ord[i-1]], cntr[ord[i]], np.max(Im)+2]
    else:
        abc = [cntr[ord[i-1]], cntr[ord[i]], cntr[ord[i+1]]]
    
    fu = fuzz.trimf(np.arange(-1, np.max(Im) + 2), abc)
    fu2 = fuzz.gaussmf(np.arange(0,256,1), lev*i, w[ord[i]] / 2)
    str1 = "ce" + str(i)
    str2 = "sl" + str(i)

    fcm_gr[str1] = fu
    fcm_sl[str2] = fu2



rl = []

for i in range(ncenters):
    s1 = "ce" + str(i)
    s2 = "sl" + str(i)
    rule = ctrl.Rule(fcm_gr[s1],fcm_sl[s2])
    rl.append(rule)

sist = ctrl.ControlSystem(rl)
sist1 = ctrl.ControlSystemSimulation(sist)

#im_gauss = np.zeros_like(Im)

Im2 = np.zeros_like(Im)


plt.subplot(121)
plt.imshow(Ima, cmap=plt.cm.gray, clim=(0, 255))
plt.title('Original')
plt.axis('off')


rt = np.zeros(257)
for j in range(0,mx + 2):
    sist1.input['grupos'] = j
    sist1.compute()
    rt[j] = sist1.output['salida']


rt[0] = 0;
rt[1] = 0;
rt[2] = 0;
rt[mx+1] = 255;
rt[mx] = 255;

    
Im2 = np.interp(Ima.flatten(), range(257), rt)

Im2 = np.reshape(Im2,[fil,col])

plt.subplot(122)
plt.axis('off')
plt.imshow(Im2,cmap=plt.cm.gray, clim=(0, 255))
plt.title('Metodo Propuesto')


plt.show()

plt.imshow(Im2, cmap='gray')
plt.axis('off')
plt.show()

input_ = np.fft.fft2(Im2)
result = ni.fourier_gaussian(input_, sigma=1)
result = np.fft.ifft2(result)
plt.imshow(result.real, cmap='gray')  # the imaginary part is an artifact
plt.show()