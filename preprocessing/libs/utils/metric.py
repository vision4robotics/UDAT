#!/usr/bin/env python
# coding: utf-8
#
import numpy as np

eps = np.finfo(np.float32).eps
def StructureMeasure(prediction, GT):
    """
        StructureMeasure computes the similarity between the foreground map and
        ground truth(as proposed in "Structure-measure: A new way to evaluate
        foreground maps" [Deng-Ping Fan et. al - ICCV 2017])
        Usage:
        Q = StructureMeasure(prediction,GT)
        Input:
        prediction - Binary/Non binary foreground map with values in the range
                        [0 1]. Type: np.float32
        GT - Binary ground truth. Type: np.bool
        Output:
        Q - The computed similarity score
    """
    # check input
    if prediction.dtype != np.float32:
        raise ValueError("prediction should be of type: np.float32")
    if np.amax(prediction) > 1 or np.amin(prediction) < 0:
        raise ValueError("prediction should be in the range of [0 1]")
    if GT.dtype != np.bool:
        raise ValueError("prediction should be of type: np.bool")

    y = np.mean(GT)

    if y == 0: # if the GT is completely black
        x = np.mean(prediction)
        Q = 1.0 - x
    elif y == 1: # if the GT is completely white
        x = np.mean(prediction)
        Q = x
    else:
        alpha = 0.5
        Q = alpha * S_object(prediction, GT) + (1 - alpha) * S_region(prediction, GT)
        if Q < 0:
            Q = 0

    return Q

def S_object(prediction, GT):
    """
        S_object Computes the object similarity between foreground maps and ground
        truth(as proposed in "Structure-measure:A new way to evaluate foreground
        maps" [Deng-Ping Fan et. al - ICCV 2017])
        Usage:
          Q = S_object(prediction,GT)
        Input:
          prediction - Binary/Non binary foreground map with values in the range
                       [0 1]. Type: np.float32
          GT - Binary ground truth. Type: np.bool
        Output:
          Q - The object similarity score
    """
    # compute the similarity of the foreground in the object level
    # Notice: inplace operation need deep copy
    prediction_fg = prediction.copy()
    prediction_fg[~GT] = 0
    O_FG = Object(prediction_fg, GT)

    # compute the similarity of the background
    prediction_bg = 1.0 - prediction;
    prediction_bg[GT] = 0
    O_BG = Object(prediction_bg, ~GT)

    # combine the foreground measure and background measure together
    u = np.mean(GT)
    Q = u * O_FG + (1 - u) * O_BG

    return Q

def Object(prediction, GT):
    # compute the mean of the foreground or background in prediction
    x = np.mean(prediction[GT])
    # compute the standard deviations of the foreground or background in prediction
    sigma_x = np.std(prediction[GT])

    score = 2.0 * x / (x * x + 1.0 + sigma_x + eps)
    return score

def S_region(prediction, GT):
    """
        S_region computes the region similarity between the foreground map and
        ground truth(as proposed in "Structure-measure:A new way to evaluate
        foreground maps" [Deng-Ping Fan et. al - ICCV 2017])
        Usage:
          Q = S_region(prediction,GT)
        Input:
          prediction - Binary/Non binary foreground map with values in the range
                       [0 1]. Type: np.float32
          GT - Binary ground truth. Type: np.bool
        Output:
          Q - The region similarity score
    """
    # find the centroid of the GT
    X, Y = centroid(GT)
    # divide GT into 4 regions
    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = divideGT(GT, X, Y)
    # Divede prediction into 4 regions
    prediction_1, prediction_2, prediction_3, prediction_4 = Divideprediction(prediction, X, Y)
    # Compute the ssim score for each regions
    Q1 = ssim(prediction_1, GT_1)
    Q2 = ssim(prediction_2, GT_2)
    Q3 = ssim(prediction_3, GT_3)
    Q4 = ssim(prediction_4, GT_4)
    #Sum the 4 scores
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def centroid(GT):
    """
        Centroid Compute the centroid of the GT
        Usage:
          X,Y = Centroid(GT)
        Input:
          GT - Binary ground truth. Type: logical.
        Output:
          X,Y - The coordinates of centroid.
    """
    rows, cols = GT.shape

    total = np.sum(GT)
    if total == 0:
        X = round(float(cols) / 2)
        Y = round(float(rows) / 2)
    else:
        i = np.arange(1, cols + 1).astype(np.float)
        j = (np.arange(1, rows + 1)[np.newaxis].T)[:,0].astype(np.float)
        X = round(np.sum(np.sum(GT, axis=0) * i) / total)
        Y = round(np.sum(np.sum(GT, axis=1) * j) / total)
    return int(X), int(Y)

def divideGT(GT, X, Y):
    """
        LT - left top;
        RT - right top;
        LB - left bottom;
        RB - right bottom;
    """
    # width and height of the GT
    hei, wid = GT.shape
    area = float(wid * hei)

    # copy 4 regions
    LT = GT[0:Y, 0:X]
    RT = GT[0:Y, X:wid]
    LB = GT[Y:hei, 0:X]
    RB = GT[Y:hei, X:wid]

    # The different weight (each block proportional to the GT foreground region).
    w1 = (X * Y) / area
    w2 = ((wid - X) * Y) / area
    w3 = (X * (hei-Y)) / area
    w4 = 1.0 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def Divideprediction(prediction, X, Y):
    """
        Divide the prediction into 4 regions according to the centroid of the GT
    """
    hei, wid = prediction.shape
    # copy 4 regions
    LT = prediction[0:Y, 0:X]
    RT = prediction[0:Y, X:wid]
    LB = prediction[Y:hei, 0:X]
    RB = prediction[Y:hei, X:wid]

    return LT, RT, LB, RB

def ssim(prediction, GT):
    """
        ssim computes the region similarity between foreground maps and ground
        truth(as proposed in "Structure-measure: A new way to evaluate foreground
        maps" [Deng-Ping Fan et. al - ICCV 2017])
        Usage:
          Q = ssim(prediction,GT)
        Input:
          prediction - Binary/Non binary foreground map with values in the range
                       [0 1]. Type: np.float32
          GT - Binary ground truth. Type: np.bool
        Output:
          Q - The region similarity score
    """
    dGT = GT.astype(np.float32)

    hei, wid = prediction.shape
    N = float(wid * hei)

    # Compute the mean of SM,GT
    x = np.mean(prediction)
    y = np.mean(dGT)

    # Compute the variance of SM,GT
    dx = prediction - x
    dy = dGT - y
    total = N - 1 + eps
    sigma_x2 = np.sum(dx * dx) / total
    sigma_y2 = np.sum(dy * dy) /  total

    # Compute the covariance between SM and GT
    sigma_xy = np.sum(dx * dy) / total

    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + eps)
    elif beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

