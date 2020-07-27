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
import random

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


# step2
def RGB2HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #
    # # for t in image:
    # #     #print(t)
    # (height, width) = image.shape[0:2]
    # out = np.ones(image.shape)
    # # out0 = [[rgb_to_hsv(y[0], y[1], y[2]) for y in x] for x in image]
    # for h in range(height):
    #     for w in range(width):
    #         out[h, w] = rgb_to_hsv(image[h, w][0], image[h, w][1], image[h, w][2])
    # tmp=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # # plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    # # plt.show()
    #
    # comparison = out == tmp
    # equal_arrays = comparison.all()
    # return out


# step2 helper
def rgb_to_hsv(r, g, b):
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)  # maximum of r, g, b
    cmin = min(r, g, b)  # minimum of r, g, b
    diff = cmax - cmin  # diff of cmax and cmin.

    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0

    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360

    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360

    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    # compute v
    v = cmax * 100
    return h, s, v


# step3 Not sure
def RGB2Gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# step4  done
def HSV_Threshold(image):
    # aH = image[:, :, 0]
    # aS = image[:, :, 1]
    # aV = image[:, :, 2]
    yr_from = np.array([20, 100, 100])
    yr_to = np.array([30, 255, 255])
    y = HSV_Thresholding(image, yr_from, yr_to)

    wr_from = np.array([0, 0, 230])
    wr_to = np.array([255, 30, 255])
    w = HSV_Thresholding(image, wr_from, wr_to)
    # yellow_threshold = cv2.inRange(image, yr_from, yr_to)
    # white_threshold = cv2.inRange(image, wr_from, wr_to)

    # plt.imshow(w)
    # plt.show()
    #
    # plt.imshow(y)
    # plt.show()

    return y, w, np.add(y, w)


# step4 helper
def HSV_Thresholding(image, rangeFrom, rangeTo):
    height, width = image.shape[0:2]
    out = np.copy(image)
    for h in range(height):
        for w in range(width):
            if (image[h, w, 0] <= rangeTo[0] and image[h, w, 1] <= rangeTo[1] and image[h, w, 2] <= rangeTo[2]) and \
                    (image[h, w, 0] >= rangeFrom[0] and image[h, w, 1] >= rangeFrom[1] and image[h, w, 2] >= rangeFrom[
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
            if np.all(mask[h, w] == 0):
                result[h, w] = 0
    return result


# step 6 done
def Gaussian(image, sigma=0.5, size=3):
    kernel_1 = [[1 / 256, 4 / 256, 6, 256, 4 / 256, 1 / 256],
                [4 / 256, 16 / 256, 24, 256, 16 / 256, 4 / 256],
                [6 / 256, 24 / 256, 36, 256, 24 / 256, 6 / 256],
                [4 / 256, 16 / 256, 24, 256, 16 / 256, 4 / 256],
                [1 / 256, 4 / 256, 6, 256, 4 / 256, 1 / 256]]
    kernel = get_gaussian_kernel(size, sigma)
    out = multiplyKernel(image, kernel)
    return out


# step 6 helper calculating Gaussian kernel
def get_gaussian_kernel(size, sigma=0.5):
    offset = size // 2
    kernel = np.empty([size, size], dtype=float)
    for x in range(size):
        for y in range(size):
            a = (sigma ** 2) * 2 * np.pi  # try sigma * sigma
            nx = x - offset
            ny = y - offset
            b = -1 * (nx ** 2 + ny ** 2)
            c = 2 * (sigma ** 2)
            kernel[x][y] = (1.0 / a) * (np.exp(b / c))

    return kernel


def multiplyKernel(image, kernel):
    kernel = np.array(kernel)

    image2 = np.pad(image, (((len(kernel[0]) // 2), (len(kernel[0]) // 2)), (((len(kernel[1]) // 2), (
            len(kernel[1]) // 2)))), 'constant')
    out = np.copy(image)
    (height, width) = image2.shape[:2]
    for h in range((len(kernel[0]) // 2), (height - (len(kernel[0]) // 2))):
        for w in range((len(kernel[1]) // 2), (width - (len(kernel[1]) // 2))):
            tmp = image2[h - (len(kernel[0]) // 2):(1 + h + (len(kernel[0]) // 2)),
                  w - (len(kernel[1]) // 2):(1 + w + (len(kernel[1]) // 2))]
            value = tmp * kernel
            out[h - (len(kernel[0]) // 2), w - (len(kernel[1]) // 2)] = np.sum(value)

    return out


# step 7
def CannyEdgeDetection(image, high, low):
    sobel, theta = Sobel(image)
    non_maximum_suppression = NonMaximumSuppression(sobel, theta)
    double_thresholding = DoubleThresholding(non_maximum_suppression, high, low)
    edge_detection = HysteresisEdgeTracking(double_thresholding)

    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].imshow(sobel, cmap='gray')
    # axarr[0, 1].imshow(non_maximum_suppression, cmap='gray')
    # axarr[1, 0].imshow(double_thresholding, cmap='gray')
    # axarr[1, 1].imshow(edge_detection, cmap='gray')
    #
    # plt.show()
    return edge_detection


# step 7 helper
def Sobel(image):
    kernel_X = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    kernel_Y = [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]

    out_X = multiplyKernel(image, kernel_X)
    out_Y = multiplyKernel(image, kernel_Y)
    # out_X[out_X == 0] = 1
    theta = np.arctan2(out_Y, out_X)

    out = np.sqrt(np.square(out_X) + np.square(out_Y))
    out = out / out.max() * 255

    return out.astype(np.uint8), theta


# step 7 helper
def NonMaximumSuppression(image, theta):
    global r
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    height, width = image.shape
    out = np.zeros(image.shape, dtype=np.uint8)

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
    for h in range(height):
        for w in range(width):
            if image[h, w] >= high_threshold:
                out[h, w] = 255
            elif high_threshold > image[h, w] >= low_threshold:
                out[h, w] = 125
            else:
                out[h, w] = 0
    return out


# step 7 helper
def HysteresisEdgeTracking(image, size=3):
    height, width = image.shape
    out = np.copy(image)

    for h in range(height):
        for w in range(width):
            if image[h, w] == 125:
                if 255 in image[(h - (size // 2)):(h + (size // 2) + 1), (w - (size // 2)):(w + (size // 2) + 1)]:
                    out[h, w] = 255
                else:
                    out[h, w] = 0
    return out


# step 9
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

            cv2.line(out, (x1, y1), (x2, y2), 255, 1)
    # plt.imshow(out, cmap='gray')
    # plt.show()

    return out


def line_detection_vectorized(image, edge_image, num_rhos=180, num_thetas=180, t_count=100):
    out = np.copy(image)
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height // 2, edge_width // 2
    #
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    #
    thetas = np.arange(-90, 90, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    #
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    #
    accumulator = np.zeros((len(rhos), len(thetas)))
    #
    # figure = plt.figure(figsize=(12, 12))
    # subplot1 = figure.add_subplot(1, 4, 1)
    # subplot1.imshow(image)
    # subplot2 = figure.add_subplot(1, 4, 2)
    # subplot2.imshow(edge_image, cmap="gray")
    # subplot3 = figure.add_subplot(1, 4, 3)
    # subplot3.set_facecolor((0, 0, 0))
    # subplot4 = figure.add_subplot(1, 4, 4)
    # subplot4.imshow(image)
    #
    edge_points = np.argwhere(edge_image != 0)
    edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])
    #
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    #
    accumulator, theta_vals, rho_vals = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas, rhos]
    )
    accumulator = np.transpose(accumulator)
    # lines = np.argwhere(accumulator > t_count)

    # for line in lines:
    #     y, x = line
    #     rho = rhos[y]
    #     theta = thetas[x]
    #     a = np.cos(np.deg2rad(theta))
    #     b = np.sin(np.deg2rad(theta))
    #     x0 = (a * rho) + edge_width_half
    #     y0 = (b * rho) + edge_height_half
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.line(out, (x1, y1), (x2, y2), 125, 1)
    #     # subplot3.plot([theta], [rho], marker='o', color="yellow")
    #     # subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

    rho1, theta1 = np.where(accumulator[:, :(len(thetas) // 2) + 1] == np.amax(accumulator[:, :(len(thetas) // 2) + 1]))
    a1 = np.cos(np.deg2rad(thetas[theta1]))
    b1 = np.sin(np.deg2rad(thetas[theta1]))
    x01 = (a1 * rhos[rho1]) + edge_width_half
    y01 = (b1 * rhos[rho1]) + edge_height_half
    x11 = int(x01 + 1000 * (-b1))
    y11 = int(y01 + 1000 * (a1))
    x21 = int(x01 - 1000 * (-b1))
    y21 = int(y01 - 1000 * (a1))

    rho2, theta2 = np.where(accumulator[:, len(thetas) // 2:] == np.amax(accumulator[:, len(thetas) // 2:]))
    theta2 += (len(thetas) // 2)
    a2 = np.cos(np.deg2rad(thetas[theta2]))
    b2 = np.sin(np.deg2rad(thetas[theta2]))
    x02 = (a2 * rhos[rho2]) + edge_width_half
    y02 = (b2 * rhos[rho2]) + edge_height_half
    x12 = int(x02 + 1000 * (-b2))
    y12 = int(y02 + 1000 * (a2))
    x22 = int(x02 - 1000 * (-b2))
    y22 = int(y02 - 1000 * (a2))

    X, Y = point_of_intersection(x11, y11, x21, y21, x12, y12, x22, y22)
    cv2.line(out, (x11, y11), (X, Y), 0, 8)
    cv2.line(out, (x12, y12), (X, Y), 255, 8)

    # subplot3.plot([theta], [rho], marker='o', color="yellow")
    # subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

    # subplot3.invert_yaxis()
    # subplot3.invert_xaxis()
    #
    # subplot1.title.set_text("Original Image")
    # subplot2.title.set_text("Edge Image")
    # subplot3.title.set_text("Hough Space")
    # subplot4.title.set_text("Detected Lines")
    # plt.show()
    # # return accumulator, rhos, thetas
    # plt.imshow(out)
    # plt.show()
    return out


def point_of_intersection(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    Ma = (Ay2 - Ay1) / (Ax2 - Ax1)
    Mb = (By2 - By1) / (Bx2 - Bx1)

    # equation1 Y=mX-mX1+Y1

    # intersection  equation1 A = equation1 B  y=y
    X = ((Ma * Ax1) / (Ma - Mb)) - ((Ay1) / (Ma - Mb)) - ((Mb * Bx1) / (Ma - Mb)) + ((By1) / (Ma - Mb))
    Y = Ma * X - Ma * Ax1 + Ay1

    return int(X), int(Y)


def main():
    path1 = 'White Lane.mp4'
    path2 = 'Yello Lane.mp4'
    path3 = 'Challenge.mp4'

    frames = ReadVideo(path1)  # step1
    out = cv2.VideoWriter('video1', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, frames[0].shape)

    for frame in frames:
        # start = datetime.now()
        hsvImg = RGB2HSV(frame)  # step2
        # print("RGB2HSV", datetime.now() - start)

        # start = datetime.now()
        grayImg = RGB2Gray(frame)  # step3
        # print("RGB2Gray", datetime.now() - start)

        # start = datetime.now()
        yellowThreshold, whiteThreshold, thresholdImg = HSV_Threshold(hsvImg)  # step4
        # print("HSVthreshold", datetime.now() - start)

        # start = datetime.now()
        mask = MaskGrayImage(grayImg, thresholdImg)  # step5

        gaussian = Gaussian(mask, 15, 3)
        canny = CannyEdgeDetection(gaussian, 200, 50)  # step7

        cannyMask = MaskGrayImage(grayImg, canny)  # step8
        houghTransform = line_detection_vectorized(frame, cannyMask)

        out.write(houghTransform)
    out.release()
    # plt.imshow(cannyMask, cmap='gray')
    # plt.imshow(canny, cmap='gray')
    # plt.show()

    # edges = cv2.Canny(gaussian2,100,200)
    # plt.imshow(edges, cmap='gray')

    # plt.imshow(gaussian2,cmap='gray')
    # f, axarr = plt.subplots(3, 2)
    #
    # axarr[0, 0].imshow(frame[0])
    # axarr[0, 1].imshow(thresholdImg, cmap='gray')
    #
    # axarr[1, 0].imshow(mask, cmap='gray')
    # axarr[1, 1].imshow(gaussian, cmap='gray')
    #
    # axarr[2, 0].imshow(canny, cmap='gray')
    # axarr[2, 1].imshow(cannyMask, cmap='gray')
    # plt.show()
    #
    # # cv2.imwrite('canny1.jpg', canny)
    # plt.imshow(cannyMask, cmap='gray')
    # plt.show()

    # print("total", datetime.now() - start0)


if __name__ == '__main__':
    main()
