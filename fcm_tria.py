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



Ima = ni.imread('cnc.jpg', flatten=True)
#plan = dicom.read_file("Q3_IMG0070.dcm")
#plan = dicom.read_file("000002.dcm")
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

#Ima = plan.pixel_array
#Ima = Ima1
fil, col = np.shape(Ima)

Im = np.reshape(Ima, [1, fil * col])

print(np.shape(Im))
start_time = time.time()
ncenters = 6;
#PE = np.zeros(ncenters)
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
        abc = [-1, cntr[ord[i]], cntr[ord[i+1]]]
        print(abc)
    elif i == ncenters-1:
        abc = [cntr[ord[i-1]], cntr[ord[i]], np.max(Im)+2]
    else:
        abc = [cntr[ord[i-1]], cntr[ord[i]], cntr[ord[i+1]]]
    #print(abc)
    fu = fuzz.trimf(np.arange(-1, np.max(Im) + 2), abc)
    print(ord[i])
    fu2 = fuzz.gaussmf(np.arange(0,256,1), lev*i, w[ord[i]] / 2)
    # fu1 = fuzz.defuzz(x, fu, 'centroid')
    str1 = "ce" + str(i)
    str2 = "sl" + str(i)

    fcm_gr[str1] = fu
    fcm_sl[str2] = fu2

#plt.subplot(132)
fcm_gr.view()
plt.show()

#plt.subplot(133)
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

#for i in range(fil*col):
    #print Im[0,i]
#    sist1.input['grupos'] = Im[0,i]
#    sist1.compute()
#    Im2[0,i] = sist1.output['salida']



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

#plt.subplot(231)
#plt.imshow(imf, cmap=plt.cm.gray)

#plt.subplot(232)
#plt.imshow(Ima, cmap=plt.cm.gray)

#plt.subplot(233)
#plt.hist(imf)
#plt.imshow(Im2, cmap=plt.cm.gray)
di = np.zeros([ncenters, mx])
mu = np.zeros([ncenters, mx])

for k in range(np.max(Im)):
    for i in range(ncenters):
        di[i, k] = np.abs(k - cntr[i])

    for i in range(ncenters):
        for j in range(ncenters):
            mu[i, k] = mu[i, k] + (di[i, k] / (di[j, k] + eps)) ** 2
        mu[i, k] = mu[i, k] ** (-1)

#plt.subplot(234)
for i in range(ncenters):
    plt.plot(range(np.max(Im)), mu[i, :])
plt.show()
# plt.subplot(235)
# plt.imshow(im_gauss, cmap=plt.cm.gray)
rt = np.zeros(mx + 1)
for j in range(0,mx + 1):
    sist1.input['grupos'] = j
    sist1.compute()
    rt[j] = sist1.output['salida']


#plt.subplot(235)
#plt.plot(range(mx+1),rt)

for i in range(fil*col):
    Im2[0,i] = rt[int(Im[0,i])]


rsmd = 0

rsmd = np.sum(Im - Im2)**2


rsmd = np.sqrt(rsmd/np.product(np.shape(Im)))

print rsmd

Im2 = np.reshape(Im2,[fil,col])



#plt.subplot(233)
#plt.imshow(Im2,cmap=plt.cm.gray)

#plt.subplot(236)
#plt.hist(Im2)

#plt.subplot(131)
plt.imshow(Im2, cmap='gray')


plt.show()

#plt.plot()
