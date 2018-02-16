from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import scipy.ndimage as ni
import time
import scipy.misc as mc

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

Ima = ni.imread('xim2.jpeg', flatten=True)



fil, col = np.shape(Ima)

Im = np.reshape(Ima, [1, fil * col])

print(np.shape(Im))
start_time = time.time()



PE = []
fcps = []

for ncenters in range(3,9):
    #ncenters = ;
    stp = 0.2 * ncenters
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


#ncenters = np.argmax(PE) + 3
#ncenters = 

print ncenters

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            Im, ncenters, 2, error=stp, maxiter=300, init=None)

print("--- %s seconds ---" % (time.time() - start_time))
# print(cntr)
# plt.subplot(321)
# plt.imshow(np.reshape(u[0,:],[fil, col]),cmap=plt.cm.gray)

# plt.subplot(322)
# plt.imshow(np.reshape(u[1,:],[fil, col]),cmap=plt.cm.gray)

# plt.subplot(323)
# plt.imshow(np.reshape(u[2,:],[fil, col]),cmap=plt.cm.gray)

# plt.subplot(324)
# plt.imshow(np.reshape(u[3,:],[fil, col]),cmap=plt.cm.gray)

cntr = np.reshape(cntr, ncenters)
ord = np.argsort(cntr)

# print(np.shape(u))

# u = u/np.sum(u, axis=0)

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
    #abc = [cntr[ord[i]] - w[ord[i]], cntr[ord[i]], cntr[ord[i]] + w[ord[i]]]
    if i == 0:
        abc = [0, 0, cntr[ord[i+1]]]
        #print(abc)
    elif i == ncenters-1:
        abc = [cntr[ord[i-1]], cntr[ord[i]], np.max(Im)+2]
    else:
        abc = [cntr[ord[i-1]], cntr[ord[i]], cntr[ord[i+1]]]
    #print(abc)
    fu = fuzz.trimf(np.arange(-1, np.max(Im) + 2), abc)
    #print(lev*i)
    fu2 = fuzz.gaussmf(np.arange(0,256,1), lev*i, w[ord[i]] / 2)
    # fu1 = fuzz.defuzz(x, fu, 'centroid')
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


plt.figure(figsize=(14 ,6))

imf, cdf = image_histogram_equalization(Ima)
imf = imf.astype(float)
plt.subplot(132)
plt.imshow(imf, cmap=plt.cm.gray, clim=(0, 255))
plt.title('Ecualizacion histograma')
plt.axis('off')
muf = np.mean(imf)
rmsef = np.sqrt(np.sum((imf-muf)**2)/(fil*col))

plt.subplot(131)
plt.imshow(Ima, cmap=plt.cm.gray, clim=(0, 255))
plt.title('Original')
plt.axis('off')


'''di = np.zeros([ncenters, mx])
mu = np.zeros([ncenters, mx])

for k in range(np.max(Im)):
    for i in range(ncenters):
        di[i, k] = np.abs(k - cntr[i])

    for i in range(ncenters):
        for j in range(ncenters):
            mu[i, k] = mu[i, k] + (di[i, k] / (di[j, k] + eps)) ** 2
        mu[i, k] = mu[i, k] ** (-1)'''

#plt.subplot(234)
#for i in range(ncenters):
#    plt.plot(range(np.max(Im)), mu[i, :])


rt = np.zeros(mx + 2)
for j in range(0,mx + 2):
    sist1.input['grupos'] = j
    sist1.compute()
    rt[j] = sist1.output['salida']


rt[1] = 0;
rt[2] = 0;
rt[mx+1] = 255;
rt[mx] = 255;

for i in range(fil*col):
    Im2[0,i] = rt[int(Im[0,i])+1]

Im2 = np.reshape(Im2,[fil,col])

plt.subplot(133)
plt.axis('off')
plt.imshow(Im2,cmap=plt.cm.gray, clim=(0, 255))
plt.title('Metodo Propuesto')
mu2 = np.mean(Im2)
rmse2 = np.sqrt(np.sum((Im2-mu2)**2)/(fil*col))

mu = np.mean(Ima)
rsmc = np.sqrt(np.sum((Ima-mu)**2)/(fil*col))

print rmsef
print rmse2
print rsmc
#print rt
plt.show()