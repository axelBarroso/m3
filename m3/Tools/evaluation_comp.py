#!/usr/bin/python
#Copyright 2015 CVC-UAB
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "Miquel Ferrarons, David Vazquez"
__copyright__ = "Copyright 2015, CVC-UAB"
__credits__ = ["Miquel Ferrarons", "David Vazquez"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Ferrarons"

import Config as cfg
import pickle
import numpy as np
from Tools import drawing
from Tools import nms
import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageChops
import matplotlib.pyplot as plt

#Computes the overlap between two bounding boxes in the format
# [TLX, TLY, BRX, BRY]
def computeOverlap(A, B):
    #Area of first bounding box
    SA = (A[3] - A[1])*(A[2] - A[0]) # width * height


    #Fixing Ratio
    h_Correction = ((B[3] - B[1]) - (B[2] - B[0]) )/2
    B[2] = B[2] + h_Correction
    B[0] = B[0] - h_Correction

    #Area of second bounding box
    SB = (B[3] - B[1])*(B[2] - B[0]) # width * height

    #Coordinates of the intersection
    Ix1 = max(A[0], B[0])
    Iy1 = max(A[1], B[1])
    Ix2 = min(A[2], B[2])
    Iy2 = min(A[3], B[3])

    Iw = Ix2 - Ix1# Width of the intersection
    Ih = Iy2 - Iy1# Height of the intersection

    SI = max(0, Iw) * max(0, Ih)#Area of the intersection

    SU = SA + SB - SI #Area of union

    if SU > 0:
        ratio = float(SI) / float(SU)
    else:
        ratio = 0.

    return ratio

# Returns Statistics (TP, FP, FN) given annotations and detections.
def evaluateImage(annotatedBoxes, detectedBoxes, modelIndexes, model, detectedBoxesScores= None): #For model compensation

    TP = 0
    FP = 0
    FN = 0

    if len(annotatedBoxes) == 0:
        FP = len(detectedBoxes)
        return TP, FP, FN
    if len(detectedBoxes) == 0:
        FN = len(annotatedBoxes)
        return TP, FP, FN

    #Sort the boxes by score, in descending order
    if detectedBoxesScores is not None:
        detectedBoxes = detectedBoxes[np.argsort(-detectedBoxesScores)]
        modelIndexes  = modelIndexes[np.argsort(-detectedBoxesScores)]


    aBoxIsAlreadyDetected = [False]*len(annotatedBoxes)
    annotationsFound = 0
    index = -1; #For window compensation
    for dBox in detectedBoxes:
        index = index + 1   #For window compensation
        #For window compensation: Ignore window if it has not been detected with the current model

        if model is not int(float(modelIndexes[index])):
            continue

        #See if it's a TP or a FP
        maxRatio = 0
        maxRatioNoSignal = 0
        for idx in range(0, len(annotatedBoxes)):
            aBox = annotatedBoxes[idx]
            currentRatio = computeOverlap(dBox, aBox)

            if aBox[4] == 1:
                if currentRatio > maxRatio:
                    maxRatio = currentRatio
            else:
                if currentRatio > maxRatioNoSignal:
                    maxRatioNoSignal = currentRatio


        if maxRatio >= maxRatioNoSignal:
            # Avoid low overlapping in a high score windows
            maxRatio = math.ceil(maxRatio*10)/10
            if maxRatio >= cfg.annotation_min_overlap:
                if not aBoxIsAlreadyDetected[idx]:
                    TP += 1
                    annotationsFound += 1
                    aBoxIsAlreadyDetected[idx] = True
                else:
                    FP += 1
            else:
                FP += 1
    #All those that were not detected are false negatives
    FN = len(annotatedBoxes) - annotationsFound
    return TP, FP, FN