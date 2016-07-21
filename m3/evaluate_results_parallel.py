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
__email__ = "miquelferrarons@gmail.com"

import Config as cfg
import pickle
import numpy as np
from Tools import nms
import os
import matplotlib.pyplot as plt
from Tools import evaluation as eval
from Tools import utils
from joblib import Parallel, delayed
import time


detection_thresholds = np.arange(cfg.decision_threshold_min,
                                     cfg.decision_threshold_max,
                                     cfg.decision_threshold_step)



def f(resultsFile):

    totalTP = np.zeros(len(detection_thresholds))
    totalFN = np.zeros(len(detection_thresholds))
    totalFP = np.zeros(len(detection_thresholds))

    resultsFilePath = cfg.resultsFolder+'/'+resultsFile

    file = open(resultsFilePath, 'r')
    imageResults = pickle.load(file)
    file.close()


    #Retrieve the data for this result
    detectedBoxes = imageResults['bboxes']
    detectedScores = imageResults['scores']
    imagePath = imageResults['imagepath']
    modelIndexes = imageResults['model']

    curThreshIDX = 0

    imageFilename = os.path.basename(imagePath) # Get the filename
    imageBasename = os.path.splitext(imageFilename)[0] #Take out the extension

    #Find the annotations for this image.
    annotationsFilePath = cfg.annotationsFolderPath+'gt.'+imageBasename+'.txt'
    annotatedBoxes = utils.readINRIAAnnotations(annotationsFilePath)

    for thresh in detection_thresholds:
        #Select only the bounding boxes that passed the current detection threshold

        # for i in range(5):
        #     idx, = np.where(modelIndexes == i)
        #     if idx == []:
        #         continue
        #     else:
        #         detectedScores[idx] = detectedScores[idx] + cfg.compensate[i]

        idx, = np.where(detectedScores > thresh)

        if len(idx) > 0:
            detectedBoxes = detectedBoxes[idx]
            detectedScores = detectedScores[idx]
            #Apply NMS on the selected bounding boxes
            detectedBoxes, detectedScores = nms.non_max_suppression_fast(detectedBoxes, detectedScores, overlapthresh= cfg.nmsOverlapThresh)
        else:
            detectedBoxes = []
            detectedScores = []

        #Compute the statistics for the current detected boxes
        TP, FP, FN = eval.evaluateImage(annotatedBoxes, detectedBoxes, detectedScores)

        totalTP[curThreshIDX] += TP
        totalFP[curThreshIDX] += FP
        totalFN[curThreshIDX] += FN

        curThreshIDX += 1

    return [totalTP, totalFP, totalFN]



def main():

    start_time = time.time()
    print 'Start evaluating results'
    fileList = os.listdir(cfg.resultsFolder)
    modelAndFeatures = '_'+'-'.join(cfg.featuresToExtract)+'_'+cfg.model
    resultsFileListModel = filter(lambda element: modelAndFeatures in element, fileList)

    resultsFileList = filter(lambda element: '.result' in element, resultsFileListModel)


    results = Parallel(n_jobs=-1)(delayed(f)(resultsFile) for resultsFile in resultsFileList)

    totalTP = np.zeros(len(detection_thresholds))
    totalFN = np.zeros(len(detection_thresholds))
    totalFP = np.zeros(len(detection_thresholds))

    for i in range(len(results)):
        for j in range(len(detection_thresholds)):
            totalTP[j] += results[i][0][j]
            totalFP[j] += results[i][1][j]
            totalFN[j] += results[i][2][j]

    #Compute metrics
    print 'totalFP'
    print totalFP

    print 'totalTP'
    print totalTP

    detection_rate = totalTP / (totalTP + totalFN) #Tasa de deteccion
    miss_rate = 1 - detection_rate #Tasa de error
    fppi = totalFP / len(resultsFileList) #FPPI (Falsos positivos por imagen)


    #Plot the results
    # plt.figure()
    # plt.plot(fppi, miss_rate, 'r', label='Miss-Rate vs FPPI')
    # plt.xlabel('FPPI ')
    # plt.ylabel('Error rate')
    #
    # plt.title(cfg.model + ' ' + cfg.modelFeatures)
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(detection_thresholds, fppi, 'r', label='Miss-Rate vs FPPI')
    # plt.xlabel('Detection Threshold')
    # plt.ylabel('FPPI')
    #
    # plt.title(cfg.model + ' ' + cfg.modelFeatures)
    # plt.legend()
    # plt.show()

    #plot the results in order to see the FScore
    precision = totalTP / (totalTP + totalFP)
    recall = totalTP / (totalTP + totalFN)
    FScore = 2 * (precision * recall) / (precision + recall)

    plt.figure()
    plt.plot(detection_thresholds,FScore, 'r', label='Miss-Rate vs FPPI')
    plt.xlabel('Detection Threshold')
    plt.ylabel('FScore')

    plt.title(cfg.model + ' ' + cfg.modelFeatures)
    plt.legend()
    plt.show()
    # print FScore


    print("Finish Process   --- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()

