import numpy as np
from icecube.photospline import spglam as glam
from icecube.photospline import splinefitstable
import os

def pad_knots(knots, order):
    """                                                                                                                                         
    Pad knots out for full support at the boundaries                                                                                            
    """
    result = []
    for k,o in zip(knots,order):
        pre = k[0] - (k[1]-k[0])*np.arange(o, 0, -1)
        post = k[-1] + (k[-1]-k[-2])*np.arange(1, o+1)
        result.append(np.concatenate((pre, k, post)))
    return result

def spline_fit(data,
               bins = None,
               range = None,
               weights = None,
               order = None,
               filename = 'spline.fits'):

    if bins is None:
        bins = data.shape[1]*[10]
        counts,bin_arrays = np.histogramdd(data,range=range,weights=weights)
        vars,bin_arrays = np.histogramdd(data,range=range,weights=weights**2)
    else:
        counts,bin_arrays = np.histogramdd(data,bins=bins,range=range,weights=weights)
        vars,bin_arrays = np.histogramdd(data,bins=bins,range=range,weights=weights**2)
    coords = [(b[1:]+b[:-1])/2. for b in bin_arrays]

    if order == None:
        order = list(np.zeros_like(bins))
    knots = pad_knots(bin_arrays, order)

    w = 1./np.sqrt(vars)
    w[~np.isfinite(w)] = np.nanmin(w)
    
    result = glam.fit(counts,w,coords,knots,order,0)
    
    if not filename is None:
        if os.path.exists(filename):
            os.system('rm '+filename)
        if filename[-5:]=='.fits':
            splinefitstable.write(result,filename)
        else:
            splinefitstable.write(result,filename+'.fits')

    return result
