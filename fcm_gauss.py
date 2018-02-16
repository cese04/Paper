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

# Ima1 = ni.imread('01_dr.JPG')
# plan = dicom.read_file("Q3_IMG0070.dcm")
plan = dicom.read_file("000002.dcm")
# Ima1 = mc.imresize(Ima1, 0.2)
# Ima1 = Ima1.astype(float)

# r = Ima1[:,:,0]
# g = Ima1[:,:,1]
# b = Ima1[:,:,2]

# su = r+g+b

# Ima = np.zeros_like(r)
# Ima = Ima.astype(float)

# Ima = np.divide(g,(su+0.001))*255
# Ima = g

Ima = plan.pixel_array
fil, col = np.shape(Ima)

Im = np.reshape(Ima, [1, fil * col])

print(np.shape(Im))
start_time = time.time()
ncenters = 3;
PE = np.zeros(ncenters)
stp = 0.1 * ncenters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    Im, ncenters, 2, error=stp, maxiter=300, init=None)

# print range(np.shape(u)[1])
pe = 0
for i in range(np.shape(u)[1]):
    for k in range(ncenters):
        pe = u[k, i] * np.log(u[k, i]) + pe

PE = -(1 / (fil * col * np.log(1 / ncenters))) * pe

print PE

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
imf = np.zeros_like(Ima)

for i in range(ncenters):
    imf = np.reshape(u[ord[i], :], np.shape(Ima)) * lev * i + imf

imf = imf / np.reshape(np.sum(u, axis=0), [fil, col])

w = np.zeros(ncenters)
for i in range(ncenters):
    w[i] = np.sum(u[i])

w = np.sqrt(w / np.sum(w) * np.max(Im))
print(w)

x = np.arange(0, 256, 1)

me = []

fcm_gr = ctrl.Antecedent(np.arange(0, np.max(Im) + 1), 'grupos')

for i in range(ncenters):
    fu = fuzz.gaussmf(np.arange(0, np.max(Im) + 1), cntr[ord[i]], w[i] / 2)
    # fu1 = fuzz.defuzz(x, fu, 'centroid')
    str1 = "ce" + str(i)
    fcm_gr[str1] = fu

fcm_gr.view()
plt.show()

im_gauss = np.zeros_like(Im)
# for i in range(len(Im)):
#    ims = 0.01
# ims2 = 0.01
#    for j in range(ncenters):
#        ims = np.fmax(u[j,i], me[j]) + ims
#        print(ims)
# ims = fuzz.interp_membership(x, me[j], u[j,i])
# ims2 = ims * ord[j]*lev + ims2
# im_gauss[i] = ims2/(ims+.001)
#    im_gauss[i] = ims

# im_gauss = np.reshape(im_gauss, [fil, col])

plt.subplot(221)
plt.imshow(imf, cmap=plt.cm.gray)

plt.subplot(222)
plt.imshow(Ima, cmap=plt.cm.gray)

plt.subplot(224)
plt.hist(imf)
di = np.zeros([ncenters, 255])
mu = np.zeros([ncenters, 255])

for k in range(255):
    for i in range(ncenters):
        di[i, k] = np.abs(k - cntr[i])

    for i in range(ncenters):
        for j in range(ncenters):
            mu[i, k] = mu[i, k] + (di[i, k] / (di[j, k] + eps)) ** 2
        mu[i, k] = mu[i, k] ** (-1)

plt.subplot(223)
for i in range(ncenters):
    plt.plot(range(255), mu[i, :])

# plt.subplot(233)
# plt.imshow(im_gauss, cmap=plt.cm.gray)

plt.show()
