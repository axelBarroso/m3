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
import time

def run():

    start_time = time.time()
    print 'Start evaluating results'
    fileList = os.listdir(cfg.resultsFolder)
    modelAndFeatures = '-'.join(cfg.featuresToExtract)+'_'+cfg.model
    resultsFileListModel = filter(lambda element: modelAndFeatures in element, fileList)

    resultsFileList = filter(lambda element: '.result' in element, resultsFileListModel)


    resultsFilePath = cfg.resultsFolder+'/'+resultsFileList[0]

    file = open(resultsFilePath, 'r')
    imageResults = pickle.load(file)
    file.close()

    predictions = imageResults['predictions']
    gt = imageResults['gt']

    print predictions
    print gt

    #Compute ponderations per class
    totalIm = 0
    compensations = np.zeros(cfg.num_classes)

    for i in range(len(gt)):
        compensations[int(gt[i])] += 1
        totalIm += 1
    compensations = compensations / totalIm



    confusionMatrix = np.zeros(shape=(cfg.num_classes,cfg.num_classes))
    recall = np.zeros(cfg.num_classes)
    precision = np.zeros(cfg.num_classes)

    for i in range(len(predictions)):
        confusionMatrix[int(predictions[i])][int(gt[i])] += 1

    for i in range(len(confusionMatrix)):
        false_negative = 0
        for j in range(len(confusionMatrix[i])):
            if i is not j:
                false_negative += confusionMatrix[j][i]
        if (confusionMatrix[i][i] + false_negative) == 0:
            recall[i] = 0
        else:
            recall[i] = (confusionMatrix[i][i] / (confusionMatrix[i][i] + false_negative)) * compensations[i]


    for i in range(len(confusionMatrix)):
        false_positives = 0
        for j in range(len(confusionMatrix)):
            if i is not j:
                false_positives += confusionMatrix[i][j]
        if (confusionMatrix[i][i] + false_positives) == 0:
                precision[i] = 0
        else:
                precision[i] = (confusionMatrix[i][i] / (confusionMatrix[i][i] + false_positives)) * compensations[i]


    averageRecall = 0
    averagePrecision = 0
    new_num_classes = cfg.num_classes


    for i in range(len(precision)):
        averagePrecision += precision[i]
        averageRecall += recall[i]
        #If there are not results for some classes, decrease number of classes to compute averages
        if precision[i] == 0 and recall[i] == 0:
            new_num_classes -= 1


    # averagePrecision = averagePrecision/new_num_classes
    # averageRecall = averageRecall/new_num_classes

    AverageFScore = 2 * (averagePrecision * averageRecall) / (averagePrecision + averageRecall)

    print "Confusion Matrix"
    print confusionMatrix

    print "Average Precision"
    print averagePrecision

    print "Average Recall"
    print averageRecall

    print "Average FScore"
    print AverageFScore

if __name__ == '__main__':
    run()

