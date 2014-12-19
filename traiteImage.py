import os
import sys
import time

import numpy
import numpy as np
import scipy.misc, scipy.io
import cPickle as pickle


def openImg(path, solidanglespath='../data/solidAngles'):
    sA = scipy.io.loadmat(solidanglespath)


    dataRaw = scipy.io.loadmat(path)
    d = dataRaw['envmap']*numpy.dstack((sA['solidAngles'], sA['solidAngles'], sA['solidAngles']))
    #d[np.isnan(d)] = 0

    dgray = d[:,:,0] * 0.25 + d[:,:,1] * 0.25 + d[:,:,2] * 0.5
    dgray[~numpy.isnan(dgray)] = dgray[~numpy.isnan(dgray)]/dgray[~numpy.isnan(dgray)].max()

    dfilter = dgray[~numpy.isnan(dgray)]
    return d, dgray, dfilter

def produceFigs(imgpath, n_couches):
    origC, origG, origF = openImg(imgpath)

    origG_display = origG.copy()
    origG_display[np.isnan(origG_display)] = 0
    scipy.misc.imsave('test_before.png', numpy.power(origG_display, 0.08))

    listHidden = [origF]

    params = []
    for i in range(n_couches):
        params.append(pickle.load(open('params_op_'+str(i)+'.txt', 'rb')))
        
        p = params[i]

        hiddenLayerVal = np.dot(listHidden[-1], p[0]) + p[1]
        listHidden.append(hiddenLayerVal)

        reconstructedImg = hiddenLayerVal
        print(i, reconstructedImg.shape)

        for j in range(len(listHidden)-2, -1, -1):
            reconstructedImg = np.dot(reconstructedImg, params[j][0].T) + params[j][2]

        origG_display = origG.copy()
        origG_display[~np.isnan(origG_display)] = reconstructedImg

        origG_display -= origG_display[~np.isnan(origG_display)].min()
        origG_display /= origG_display[~np.isnan(origG_display)].max()
        origG_display[np.isnan(origG_display)] = 0

        
        scipy.misc.imsave('test_layer_'+str(i+1)+'.png', numpy.power(origG_display, 0.6))


if __name__ == '__main__':
    produceFigs("../data/20130823/20130823_120323_envmap.exr.mat", 3)
    