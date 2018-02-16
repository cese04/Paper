from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import scipy.ndimage as ni
import time
import scipy.misc as mc
import dicom

eps = 0.0001

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized = np.reshape(image_equalized,image.shape)
    return image_equalized.reshape(image.shape), cdf



#plan = dicom.read_file("Q3_IMG0070.dcm")
plan = dicom.read_file("000001.dcm")


Ima = plan.pixel_array

fil, col = np.shape(Ima)

Im = np.reshape(Ima, [1, fil * col])

print(np.shape(Im))
start_time = time.time()
ncenters = 5;
PE = np.zeros(ncenters)
stp = 0.1 * ncenters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    Im, ncenters, 2, error=stp, maxiter=300, init=None)


pe = 0
for i in range(np.shape(u)[1]):
    for k in range(ncenters):
        pe = u[k, i] * np.log(u[k, i]) + pe

PE = -(1 / (fil * col * np.log(1 / ncenters))) * pe

print PE

print("--- %s seconds ---" % (time.time() - start_time))


cntr = np.reshape(cntr, ncenters)
ord = np.argsort(cntr)



lev = (255) / (ncenters - 1)
imf = np.zeros_like(Ima)

'''for i in range(ncenters):
    imf = np.reshape(u[ord[i], :], np.shape(Ima)) * lev * i + imf

imf = imf / np.reshape(np.sum(u, axis=0), [fil, col])'''

mx = np.max(Im)
mx = mx.astype(int)

imf, cdf = image_histogram_equalization(Ima, number_bins=mx)

w = np.zeros(ncenters)
for i in range(ncenters):
    w[i] = np.sum(u[i])

w = np.sqrt(w / np.sum(w) * 255)*6
print(w)



me = []

print mx
fcm_gr = ctrl.Antecedent(np.arange(-1, np.max(Im) + 2), 'grupos')
fcm_sl = ctrl.Consequent(np.arange(0,256,1), 'salida')
for i in range(ncenters):

    if i == 0:
        abc = [0, 0, cntr[ord[i+1]]]
        print(abc)
    elif i == ncenters-1:
        abc = [cntr[ord[i-1]], cntr[ord[i]], np.max(Im)+2]
    else:
        abc = [cntr[ord[i-1]], cntr[ord[i]], cntr[ord[i+1]]]

    fu = fuzz.trimf(np.arange(-1, np.max(Im) + 2), abc)
    print(ord[i])
    fu2 = fuzz.gaussmf(np.arange(0, 256), lev*i, w[ord[i]] / 2)

    str1 = "ce" + str(i)
    str2 = "sl" + str(i)

    fcm_gr[str1] = fu
    fcm_sl[str2] = fu2

fcm_gr.view()
plt.show()

fcm_sl.view()
plt.show()

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



plt.subplot(231)
plt.imshow(imf, cmap=plt.cm.gray, clim=(0, 255))

plt.subplot(232)
plt.imshow(Ima, cmap=plt.cm.gray, clim=(0, mx))


di = np.zeros([ncenters, mx])
mu = np.zeros([ncenters, mx])

for k in range(np.max(Im)):
    for i in range(ncenters):
        di[i, k] = np.abs(k - cntr[i])

    for i in range(ncenters):
        for j in range(ncenters):
            mu[i, k] = mu[i, k] + (di[i, k] / (di[j, k] + eps)) ** 2
        mu[i, k] = mu[i, k] ** (-1)

plt.subplot(234)
for i in range(ncenters):
    plt.plot(range(np.max(Im)), mu[i, :])

# Calcular para cada valor posible
rt = np.zeros(mx + 2)
for j in range(0,mx + 2):
    sist1.input['grupos'] = j
    sist1.compute()
    rt[j] = sist1.output['salida']


plt.subplot(235)
plt.plot(range(mx+2),rt)

# Obtener el valor para cada pixel de la imagen original
for i in range(fil*col):
    Im2[0,i] = rt[int(Im[0,i])+1]




#rsmd = np.sum(Im - Im2)**2


#rsmd = np.sqrt(rsmd/np.product(np.shape(Im)))

#print rsmd

Im2 = np.reshape(Im2,[fil,col])

plt.subplot(233)
plt.imshow(Im2,cmap=plt.cm.gray, clim=(0, 255))

plt.subplot(236)
plt.hist(Im2)


plt.show()