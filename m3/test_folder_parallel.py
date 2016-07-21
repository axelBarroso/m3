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

import os
from Tools import feature_extractor
import pickle
import numpy as np
from skimage import io
from skimage.util import pad
from skimage.transform import pyramid_gaussian
from skimage.util.shape import view_as_windows
import skimage.util as util
from joblib import Parallel, delayed
import random
from PIL import Image
import math
from Tools import nms
from Tools import utils
import Config as cfg
import time
from skimage.transform import resize


def f(filename):

    inputfolder = cfg.testFolderPath
    outputfolder = cfg.resultsFolder
    decisionThreshold = cfg.decision_threshold

    applyNMS= False
    # applyNMS= True #For bootstrapping and Crops

    imagepath = inputfolder + '/' + filename
    print 'Processing '+imagepath

    #Test the current image
    bboxes, scores, indices = testImage(imagepath, decisionThreshold, applyNMS)

    #Store the result in a dictionary
    result = dict()
    result['imagepath'] = imagepath
    result['bboxes'] = bboxes
    result['scores'] = scores
    result['model'] = indices

    #Save the features to a file using pickle
    outputFile = open(outputfolder+'/'+filename+'_'+'-'.join(cfg.featuresToExtract)+'_'+cfg.model+'.results', "wb")
    pickle.dump(result, outputFile)
    outputFile.close()


def main():

    print "STARTING PARALLELISM"
    start_time = time.time()
    inputfolder = cfg.testFolderPath
    fileList = os.listdir(inputfolder)
    imagesList = filter(lambda element: '.jpg' in element, fileList)

    #bootstrapping: Create boostrapping folder
    # if not os.path.exists("Bootstrapping"):
    #     os.makedirs("Bootstrapping")
    #Crops: Create Crops folder
    # if not os.path.exists("Crops"):
    #     os.makedirs("Crops")

    print 'Start processing '+inputfolder

    Parallel(n_jobs=3)(delayed(f)(filename) for filename in imagesList)
    print("Finish Process   --- %s seconds ---" % (time.time() - start_time))


def testImage(imagePath, decisionThreshold, applyNMS):

    fileList = os.listdir(cfg.modelRootPath)

    # Filter all model files
    modelsList = filter(lambda element: '.model' in element, fileList)

    # Filter our specific feature method
    currentModel = cfg.model+'_'+cfg.modelFeatures
    currentModelsList = filter(lambda element: currentModel in element, modelsList)

    models = []
    subImages = [] #To save backgorund crops

    for modelname in currentModelsList:

        file = open(cfg.modelRootPath + modelname, 'r')
        svc = pickle.load(file)
        models.append(svc)

        file.close()

    image = io.imread(imagePath, as_grey=True)
    image = util.img_as_ubyte(image) #Read the image as bytes (pixels with values 0-255)

    rows, cols = image.shape
    pyramid = tuple(pyramid_gaussian(image, downscale=cfg.downScaleFactor))

    scale = 0
    boxes = None
    scores = None
    indices = None


    for p in pyramid[0:]:
        #We now have the subsampled image in p
        window_shape = (32,32)

        #Add padding to the image, using reflection to avoid border effects
        if cfg.padding > 0:
            p = pad(p,cfg.padding,'reflect')

        try:
            views = view_as_windows(p, window_shape, step=cfg.window_step)
        except ValueError:
            #block shape is bigger than image
            break

        num_rows, num_cols, width, height = views.shape

        for row in range(0, num_rows):
            for col in range(0, num_cols):
                #Get current window
                subImage = views[row, col]
                # subImages.append(subImage)   #To save backgorund crops: Accumulate them in an array
                #Extract features
                feats = feature_extractor.extractFeatures(subImage)

                #Obtain prediction score for each model
                for model in models:

                    decision_func = model.decision_function(feats)
                    idx = models.index(model)
                    decision_func += cfg.compensate[idx]
                    if decision_func > decisionThreshold:
                    # if decision_func > -0.2:  #For bootstrapping
                        # Signal found!
                        h, w = window_shape
                        scaleMult = math.pow(cfg.downScaleFactor, scale)

                        x1 = int(scaleMult * (col*cfg.window_step - cfg.padding + cfg.window_margin))
                        y1 = int(scaleMult * (row*cfg.window_step - cfg.padding + cfg.window_margin))
                        x2 = int(x1 + scaleMult*(w - 2*cfg.window_margin))
                        y2 = int(y1 + scaleMult*(h - 2*cfg.window_margin))

                        if(y1 > 0) and (y2 > 0):
                            if y2 - y1 > 330:
                                continue

                        #bootstrapping: Save image (if positive)
                        # print(decision_func)
                        # subImages.append(subImage)

                        bbox = (x1, y1, x2, y2)
                        score = decision_func[0]

                        if boxes is not None:
                            boxes = np.vstack((bbox, boxes))
                            scores = np.hstack((score, scores))
                            indices = np.hstack((idx, indices))
                        else:
                            boxes = np.array([bbox])
                            scores = np.array([score])
                            indices = np.array([idx])
                        break

        scale += 1


    # To save backgorund crops
    # numSubImages = len(subImages)
    # for x in range(0,10): #Save 10 crops for each background image
    #     randomIndex = random.randint(1,numSubImages-1) #Get a random window index
    #     imageName = imagePath.split('/')  #Working on the crop name...
    #     imageName = imageName[len(imageName)-1]
    #     filename = (imageName[:-4]+'-'+str(x)+'.jpg')
    #     io.imsave('Results/'+filename, subImages[randomIndex])  #Save the crop
    #end To save backgorund crops

    # To save bootstrapping windows
    # numSubImages = len(subImages)
    # length = min(10, len(subImages))
    # if length > 0:
    #     for x in range(0,length) : #Save windows with detections (max 10)
    #         if numSubImages == 1:
    #             randomIndex = 0
    #         else:
    #             randomIndex = random.randint(1, numSubImages-1) #Get a random window index
    #         imageName = imagePath.split('/')  #Working on the crop name...
    #         imageName = imageName[len(imageName)-1]
    #         filename = (imageName[:-4]+'-'+str(x)+'.jpg')
    #         io.imsave('Bootstrapping/'+filename, subImages[randomIndex])  #Save the crop
    #end To save bootstrapping windows


    if applyNMS:
        #From all the bounding boxes that are overlapping, take those with maximum score.
        boxes, scores = nms.non_max_suppression_fast(boxes, scores, cfg.nmsOverlapThresh)

    #Color boostraping
    # save_windows(boxes, imagePath)

    return boxes, scores, indices



def save_windows(boxes, imagePath):
    image_color = io.imread(imagePath, as_grey=False)
    image_color = util.img_as_ubyte(image_color)
    imageFilename = os.path.basename(imagePath) # Get the filename
    imageBasename = os.path.splitext(imageFilename)[0] #Take out the extension
    annotationsFilePath = cfg.annotationsFolderPath+'gt.'+imageBasename+'.txt'
    annotatedBoxes = utils.readINRIAAnnotations(annotationsFilePath)
    signalTypes = utils.readINRIAAnnotationsDetection(annotationsFilePath)
    signalTypes = list(reversed(signalTypes))
    count = 0
    for box in boxes:
        if box[0] < 0 or box[1] < 0:
            continue
        if box[2] >= image_color.shape[1].__int__() or \
                        box[3] >= image_color.shape[0].__int__():
            continue
        annotated = 'NONSIGNAL'
        for idx in range(0, len(annotatedBoxes)):
            aBox = annotatedBoxes[idx]
            currentRatio = computeOverlap(box, aBox)
            currentRatio = math.ceil(currentRatio*10)/10
            if currentRatio > 0.5:
                annotated = signalTypes[idx]
                break
        crop = image_color[box[1]:box[3],box[0]:box[2]]
        imageName = imagePath.split('/')  #Working on the crop name...
        fileName = imageName[len(imageName)-1]
        fileName = fileName[:len(fileName)-4]
        fileName = (fileName+'.'+str(count))
        filename = (fileName+'.'+annotated+'.jpg')
        crop = resize(crop,(32,32))
        io.imsave('Crops/'+filename, crop)  #Save the crop
        print('Crop saved')
        count += 1

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


if __name__ == "__main__":
    main()