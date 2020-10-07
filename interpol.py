# This module contains some educational interpolation routines. The
# implementations are focused on illustrating the mathematical algorithm, rather
# than on efficiency or robustness, and therefore their performance on large
# data sets may be poor.

import numpy as np

def linear(xq, xp, yp):
    """Piecewise-linear interpolation of one-dimensional data.

    Parameters
    ----------
    xq : scalar or ndarray vector
        The x-coordinates at which to evaluate the inteprolated values.
    xp : ndarray vector
        The x-coordinates of the data points; must be increasing.
    yp : ndarray vector
        The y-coordinates of the data points; must be same length as xp.

    Returns
    -------
    yq : scalar or ndarray vector
        The interpolated value(s); same shape as xq.
    """

    # Input control
    xq = np.array(xq, ndmin=1) # now it's ndarray vector for sure
    xp, yp = np.array(xp), np.array(yp) # now they are ndarray for sure
    assert len(xp.shape) == 1, "Data x- and y-coordinates musy be vectors."
    assert np.all(np.diff(xp) > 0), "Data x-coordinates must be increasing."
    assert np.shape(yp) == np.shape(xp), "Data x and y must be same length."

    # Run piecewise linear interpolation on each element of xq
    yq = np.nan*np.ones_like(xq)
    for k in range(xq.size):
        x = xq[k]
        if x < xp[0] or x > xp[-1]:
            continue # leave yq[k] as NaN

        for ind in range(xp.size): # search for bracketing interval
            if xp[ind] > x:
                break
        ind = ind - 1

        yq[k] = (yp[ind] +
            (yp[ind+1] - yp[ind])/(xp[ind+1] - xp[ind])*(x - xp[ind]))

    # ABYU
    if yq.size == 1:
        yq = yq[0]
    return yq

def lagrange_cubic(xq, xp, yp):
    """Piecewise-cubic interpolation with Lagrange polynomials.

    Parameters
    ----------
    xq : scalar or ndarray vector
        The x-coordinates at which to evaluate the inteprolated values.
    xp : ndarray vector
        The x-coordinates of the data points; must be increasing.
    yp : ndarray vector
        The y-coordinates of the data points; must be same length as xp.

    Returns
    -------
    yq : scalar or ndarray vector
        The interpolated value(s); same shape as xq.
    """

    # Input control
    xq = np.array(xq, ndmin=1) # now it's ndarray vector for sure
    xp, yp = np.array(xp), np.array(yp) # now they are ndarray for sure
    assert len(xp.shape) == 1, "Data x- and y-coordinates musy be vectors."
    assert np.all(np.diff(xp) > 0), "Data x-coordinates must be increasing."
    assert np.shape(yp) == np.shape(xp), "Data x and y must be same length."

    # Run piecewise cubic interpolation on each element of xq
    yq = np.nan*np.ones_like(xq)
    for k in range(xq.size):
        x = xq[k]
        if x < xp[1] or x >= xp[-2]: # leave yq[k] as NaN
            continue

        # search for bracketing interval
        for ind in range(xp.size):
            if xp[ind] > x:
                break
        ind = ind - 1

        x1, y1 = xp[ind-1], yp[ind-1]
        x2, y2 = xp[ind+0], yp[ind+0]
        x3, y3 = xp[ind+1], yp[ind+1]
        x4, y4 = xp[ind+2], yp[ind+2]

        f1 = ((x - x2)*(x - x3)*(x - x4))/((x1 - x2)*(x1 - x3)*(x1 - x4))
        f2 = ((x - x1)*(x - x3)*(x - x4))/((x2 - x1)*(x2 - x3)*(x2 - x4))
        f3 = ((x - x1)*(x - x2)*(x - x4))/((x3 - x1)*(x3 - x2)*(x3 - x4))
        f4 = ((x - x1)*(x - x2)*(x - x3))/((x4 - x1)*(x4 - x2)*(x4 - x3))

        yq[k] = y1*f1 + y2*f2 + y3*f3 + y4*f4

    # ABYU
    if yq.size == 1:
        yq = yq[0]
    return yq

def pixelfix(I, pxq, copy=True):
    """Use bilinear interpolation to replace pixels in image.

    Parameters
    ----------
    I : ndarray, M-by-N-by-3
        A color image array to fix, in RGB format.
    pxq : ndarray, P-by-2
        List of pixel indices to replace.
    copy : bool, optional
        If True (default), return a copy of the image with replaced pixels;
        otherwise fix pixles in-place.

    Returns
    -------
    Iq : ndarray, same shape as I
        Copy of image with fixed pixels; only if copy==True.
    """
    Iq = I
    x = y = red = green = blue = 0
    for i in pxq:
        x = i[0]
        y = i[1]
        if x == 0:
            if y == 0:
                red = (I[x+1, y, 0] + I[x, y+1, 0])/2
                green = (I[x+1, y, 1] + I[x, y+1, 1])/2
                blue = (I[x+1, y, 2] + I[x, y+1, 2])/2
            elif y == 719:
                red = (I[x+1, y, 0] + I[x, y-1, 0])/2
                green = (I[x+1, y, 1] + I[x, y-1, 1])/2
                blue = (I[x+1, y, 2] + I[x, y-1, 2])/2
            else:
                red = (I[x+1, y, 0] + I[x, y+1, 0] + I[x, y-1, 0])/3
                green = (I[x+1, y, 1] + I[x, y+1, 1] + I[x, y-1, 1])/3
                blue = (I[x+1, y, 2] + I[x, y+1, 2] + I[x, y-1, 2])/3
        elif x == 719:
            if y == 0:
                red = (I[x-1, y, 0] + I[x, y+1, 0])/2
                green = (I[x-1, y, 1] + I[x, y+1, 1])/2
                blue = (I[x-1, y, 2] + I[x, y+1, 2])/2
            elif y == 719:
                red = (I[x-1, y, 0] + I[x, y-1, 0])/2
                green = (I[x-1, y, 1] + I[x, y-1, 1])/2
                blue = (I[x-1, y, 2] + I[x, y-1, 2])/2
            else:
                red = (I[x-1, y, 0] + I[x, y+1, 0] + I[x, y-1, 0])/3
                green = (I[x-1, y, 1] + I[x, y+1, 1] + I[x, y-1, 1])/3
                blue = (I[x-1, y, 2] + I[x, y+1, 2] + I[x, y-1, 2])/3
        else:
            if y == 0:
                red = (I[x-1, y, 0] + I[x, y+1, 0] + I[x+1, y, 0])/3
                green = (I[x-1, y, 1] + I[x, y+1, 1] + I[x+1, y, 1])/3
                blue = (I[x-1, y, 2] + I[x, y+1, 2] + I[x+1, y, 2])/3
            elif y == 719:
                red = (I[x-1, y, 0] + I[x+1, y, 0] + I[x, y-1, 0])/3
                green = (I[x-1, y, 1] + I[x+1, y, 1] + I[x, y-1, 1])/3
                blue = (I[x-1, y, 2] + I[x+1, y, 2] + I[x, y-1, 2])/3
            else:
                red = (I[x+1, y, 0] + I[x, y+1, 0] + I[x, y-1, 0] + I[x-1, y, 0])/4
                green = (I[x+1, y, 1] + I[x, y+1, 1] + I[x, y-1, 1] + I[x-1, y, 1])/4
                blue = (I[x+1, y, 2] + I[x, y+1, 2] + I[x, y-1, 2] + I[x-1, y, 2])/4
        Iq[x, y, 0] = red
        Iq[x, y, 1] = green
        Iq[x, y, 2] = blue
    return Iq
