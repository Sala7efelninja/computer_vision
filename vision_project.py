# main part

# 1 read the image
# 2 convert to HSV
# 3 convert to Gray
# 4 Threshold HSV for Yellow and White (combine the two results together)
# 5 Mask the gray image using the threshold output fro step 4
# 6 Apply noise remove (gaussian) to the masked gray image
# 7 use canny detector and fine tune the thresholds (low and high values)
# 8 mask the image using the canny detector output
# 9 apply hough transform to find the lanes
# 10 apply the pipeline you developed to the challenge videos
# 11 You should submit your code

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import math
import cv2
from scipy import interpolate


# step1 done
def ReadVideo(path):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    frames = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        success, image = vidcap.read()

    return frames


# step2 Not sure
def RGB2HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    (height, width) = image.shape[0:2]
    out = np.ones(image.shape)
    for h in range(height):
        for w in range(width):
            out[h, w] = rgb_to_hsv(image[h, w][0], image[h, w][1], image[h, w][2])
    return out


# step2 helper ( not used)
def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return h, s, v


# step3 Not sure
def RGB2Gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# step4  done
def HSV_Threshold(image):

    aH=image[:,:,0]
    aS=image[:,:,1]
    aV=image[:,:,2]
    yr_from = np.array([20, 100, 100])
    yr_to = np.array([30, 255, 255])
    y = HSV_Thresholding(image, yr_from, yr_to)
    
    wr_from = np.array([0, 50, 230])
    wr_to = np.array([255, 50, 255])
    w = HSV_Thresholding(image, wr_from, wr_to)
    # yellow_threshold = cv2.inRange(image, yr_from, yr_to)
    # white_threshold = cv2.inRange(image, wr_from, wr_to)
    plt.imshow(w)
    
    
    return y, w, np.add(y, w)


# step4 helper
def HSV_Thresholding(image, rangeFrom, rangeTo): 
    height, width = image.shape[0:2]
    out = np.zeros(image.shape)
    for h in range(height):
        for w in range(width):
            if (image[h, w, 0] < rangeTo[0] and image[h, w, 1] < rangeTo[1] and image[h, w, 2] < rangeTo[2]) and \
                    (image[h, w, 0] > rangeFrom[0] and image[h, w, 1] > rangeFrom[1] and image[h, w, 2] > rangeFrom[
                        2]):
                out[h, w] = [255, 255, 255]
            else:
                out[h, w] = [0, 0, 0]
    return out.astype(int)


# step5 not sure
def MaskGrayImage(image, mask):
    result = np.copy(image)
    (height, width) = image.shape[:2]
    # result[mask == 255] = 0
    for h in range(height):
        for w in range(width):
            if mask[h, w].all() == 255:
                result[h, w] = 255
    return result


# step 6 done
def Gaussian(image, sigma=0.5):
    kernel_1 = [[1 / 256, 4 / 256, 6, 256, 4 / 256, 1 / 256],
                [4 / 256, 16 / 256, 24, 256, 16 / 256, 4 / 256],
                [6 / 256, 24 / 256, 36, 256, 24 / 256, 6 / 256],
                [4 / 256, 16 / 256, 24, 256, 16 / 256, 4 / 256],
                [1 / 256, 4 / 256, 6, 256, 4 / 256, 1 / 256]]
    kernel = get_gaussian_kernel(sigma)
    out = multiplyKernel(image, kernel)
    return out


# step 6 helper calculating Gaussian kernel
def get_gaussian_kernel(sigma=0.5):
    if sigma > 1:
        size = 5
    else:
        size = 3

    offset = size // 2
    kernel = np.empty([size, size], dtype=float)
    for x in range(size):
        for y in range(size):
            a = (sigma ** 2) * 2 * np.pi
            nx = x - offset
            ny = y - offset
            b = -1 * (nx ** 2 + ny ** 2)
            c = 2 * (sigma ** 2)
            kernel[x][y] = (1.0 / a) * (np.exp(b / c))

    return kernel


def multiplyKernel(image, kernel):
    kernel = np.array(kernel)

    image2 = np.pad(image, (((len(kernel[0] // 2)), (len(kernel[0] // 2))), (((len(kernel[1] // 2)), (
        len(kernel[1] // 2))))), 'constant')
    out = np.copy(image)
    (height, width) = image2.shape[:2]
    for h in range(len(kernel[0] // 2), (height - len(kernel[0] // 2))):
        #print(h)
        for w in range(len(kernel[1] // 2), (width - len(kernel[1] // 2))):
            tmp = image2[h - (len(kernel[0]) // 2):(1 + h + (len(kernel[0]) // 2)),
                  w - (len(kernel[1]) // 2):(1 + w + (len(kernel[1]) // 2))]
            value = tmp * kernel
            out[h - len(kernel[0] // 2), w - len(kernel[1] // 2)] = np.sum(value)

    return out


# step 7
def CannyEdgeDetection(image,low,high):
    sobel, theta = Sobel(image)
    non_maximum_suppression = NonMaximumSuppression(sobel, theta)
    double_thresholding = DoubleThresholding(non_maximum_suppression, high, low)
    edge_detection = HysteresisEdgeTracking(double_thresholding)

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(sobel, cmap='gray')
    axarr[0, 1].imshow(non_maximum_suppression, cmap='gray')
    axarr[1, 0].imshow(double_thresholding, cmap='gray')
    axarr[1, 1].imshow(edge_detection, cmap='gray')

    plt.show()
    return edge_detection


# step 7 helper
def Sobel(image):
    kernel_X = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    kernel_Y = [[1, 2, 1],
                [-0, 0, 0],
                [-1, -2, -1]]

    out_X = multiplyKernel(image, kernel_X)
    out_Y = multiplyKernel(image, kernel_Y)
    # out_X[out_X == 0] = 1
    theta = np.arctan2(out_Y, out_X)

    out = np.sqrt(np.square(out_X) + np.square(out_Y))
    out *= 255 / out.max()

    return out.astype(np.uint8), theta


# step 7 helper
def NonMaximumSuppression(image, theta):
    global r
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    height, width = image.shape
    out = np.zeros((height, width), dtype=np.uint8)

    for h in range(1, height - 1):
        for w in range(1, width - 1):

            if (0 <= angle[h, w] < 22.5) or (157.5 <= angle[h, w] <= 180):
                q = image[h, w + 1]
                r = image[h, w - 1]
                # angle 45
            elif 22.5 <= angle[h, w] < 67.5:
                q = image[h + 1, w + 1]
                r = image[h - 1, w - 1]
                # angle 90
            elif 67.5 <= angle[h, w] < 112.5:
                q = image[h + 1, w]
                r = image[h - 1, w]
                # angle 135
            elif 112.5 <= angle[h, w] < 157.5:
                q = image[h - 1, w + 1]
                r = image[h + 1, w - 1]

            if q <= image[h, w] >= r:
                out[h, w] = image[h, w]
            else:
                out[h, w] = 0
    return out


# step 7 helper
def DoubleThresholding(image, high_threshold, low_threshold):
    height, width = image.shape
    out = np.copy(image)
    for h in range(height - 1):
        for w in range(width - 1):
            if image[h, w] >= high_threshold:
                out[h, w] = 255
            elif high_threshold >= image[h, w] >= low_threshold:
                out[h, w] = 25
            else:
                out[h, w] = 0
    return out


# step 7 helper
def HysteresisEdgeTracking(image, size=3):
    height, width = image.shape
    out = np.copy(image)
    for h in range(height):
        for w in range(width):
            if out[h, w] == 25:
                if out[(h - (size // 2)):(h + (size // 2) + 2), (w - (size // 2)):(w + (size // 2) + 2)].any == 255:
                    out[h, w] = 255
                else:
                    out[h, w] = 0
    return out


# step 8
def HoughTransofrm(image):
    minLineLength = 150
    # maxLineGap = 5
    # out = np.zeros(image.shape)
    # lines = cv2.HoughLinesP(image, 10, np.pi / 180, 400, minLineLength, maxLineGap)
    # for i in range(0, len(lines)):
    #     for x1, y1, x2, y2 in lines[i]:
    #         cv2.line(out, (x1, y1), (x2, y2), 255, 2)
    # plt.imshow(out, cmap='gray')
    # plt.show()

    out = np.zeros(image.shape)
    # edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(image, 0.5, np.pi / 180, 30)
    for x in range(0, len(lines)):
        for rho, theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(out, (x1, y1), (x2, y2), 255, 8)
    plt.imshow(out, cmap='gray')
    plt.show()

    return out

def main():
    path1 = 'White Lane.mp4'
    path2 = 'Yello Lane.mp4'
    path3 = 'Challenge.mp4'

    frame = ReadVideo(path3)
    hsvImg = RGB2HSV(frame[0])
    grayImg = RGB2Gray(frame[0])
    yellowThreshold, whiteThreshold, thresholdImg = HSV_Threshold(hsvImg)
    mask1 = MaskGrayImage(grayImg, thresholdImg)
    
#    gaussian1 = Gaussian(mask1, 1.5)
#    gaussian2 = Gaussian(mask1, 1)
#    gaussian3 = Gaussian(mask1, 0.5)
    gaussian4 = Gaussian(mask1, 3)
    
    canny = CannyEdgeDetection(gaussian4,50,60)
    cannyMask = MaskGrayImage(gaussian2, canny)
    houghTransform = HoughTransofrm(cannyMask)
    plt.imshow(cannyMask, cmap='gray')
    plt.imshow(canny, cmap='gray')
    plt.show()
    
    #edges = cv2.Canny(gaussian2,100,200)
    #plt.imshow(edges, cmap='gray')
    
    #plt.imshow(gaussian2,cmap='gray')
    # f, axarr = plt.subplots(3, 2)
    #
    # axarr[0, 0].imshow(frame[0])
    # axarr[0, 1].imshow(hsvImg)
    #
    # axarr[1, 0].imshow(thresholdImg)
    # axarr[1, 1].imshow(mask)
    #
    # axarr[2, 0].imshow(gaussian)
    # axarr[2, 1].imshow(Gaussian(mask, 1.4))

    # cv2.imwrite('canny1.jpg', canny)
    # plt.imshow(edges, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main()
