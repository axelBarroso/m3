
from skimage import color


def colorSegmentation(img):

    image_HSV = color.rgb2hsv(img)

    imageHue = image_HSV[:,:,0]

    for i in range(len(imageHue[:,1])):
        for j in range(len(imageHue[1,:])):
            if imageHue[i][j] > 0.5 and imageHue[i][j] < 0.7:
                imageHue[i][j] = 1
            elif imageHue[i][j] < 0.15:
                imageHue[i][j] = 1
            else:
                imageHue[i][j] = 0

    return imageHue
