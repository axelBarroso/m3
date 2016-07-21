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

import numpy as np
# Reads the annotations in the INRIA format [XC, YC, W, H]
#   XC: Center of the pedestrian in X
#   YC: Center of the pedestrian in Y
#    W: Width of the window
#    H: Height of the window
#
# And returns the bounging boxes as [TLx,TLy,BRx,BRy]
#   TLx :Top-Left X,
#   TLy :Top-Left Y,
#   BRx :Bottom-Right X,
#   BRy :Bottom-Right y,

signalsToDetect = ['A13', 'A14', 'A15', 'A1A', 'A1B', 'A1C', 'A1D', 'A23', 'A25', 'A29', 'A30', 'A33', 'A41', 'A51', 'A7A', 'A7B', 'A7C'
                   , 'B15A', 'B17', 'B1', 'B19', 'B5', 'B21', 'B11', 'B9'
                   , 'C1', 'C11', 'C21', 'C23', 'C27', 'C29', 'C3', 'C31LEFT', 'C31RIGHT', 'C35', 'C43'
                   , 'D1a', 'D10', 'D1b', 'D3b', 'D5', 'D7', 'D9'
                   , 'E1', 'E3', 'E5', 'E7' 'E9a', 'E9a_miva', 'E9b', 'E9c', 'E9d', 'E9e'
                   , 'F12a', 'F12b', 'F45', 'F47', 'F49', 'F50', 'F59', 'F87']

def readINRIAAnnotations(annotationsPath):

    annotatedBoxes = None

    with open(annotationsPath,'r') as fp:
        for line in fp:


            #Inria annotates the center of the pedestrian, and the width and height
            # xc, yc, w, h, string = line.split(' ')
            # x1 = float(xc) - float(w)/2
            # y1 = float(yc) - float(h)/2
            # x2 = float(x1) + float(w)
            # y2 = float(y1) + float(h)


            # x1, y1, x2, y2, string, string = line.split(' ')
            text = line.split(' ')
            x1 = text[0]
            y1 = text[1]
            x2 = text[2]
            y2 = text[3]
            signalType = text[4].split('\n')
            print signalType[0]
            if signalType[0] in signalsToDetect:
                bbox = (float(y1), float(x1), float(y2), float(x2), 1.)
            else:
                bbox = (float(y1), float(x1), float(y2), float(x2), 0.)

            if annotatedBoxes is not None:
                annotatedBoxes = np.vstack((bbox, annotatedBoxes))
            else:
                annotatedBoxes = np.array([bbox])

    return annotatedBoxes
