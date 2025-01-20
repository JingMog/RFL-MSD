import os
import cv2
import numpy
import random

#img [h, w, c] in [0,255]
def Generate_PadLine(img):
    # img = img.transpose([1, 2, 0])
    h = img.shape[0]
    w = img.shape[1]
    # w_p = random.randint(0, min(w, 80))
    # h_p = random.randint(0, 16)
    w_p = random.randint(0, min(w, 60))
    h_p = random.randint(0, 12)
    x0 = int(random.random() * w)
    y0 = int(random.random() * h)
    if x0 + w_p < w:
        row_s = x0
        row_e = x0 + w_p
    else:
        row_s = x0 - w_p
        row_e = x0
    if y0 + h_p < h:
        col_s = y0
        col_e = y0 + h_p
    else:
        col_s = y0 - h_p
        col_e = y0
    img[0:h, row_s:row_e, :] = [0, 0, 0]
    img[col_s:col_e, 0:w, :] = [0, 0, 0]
    # img = img.transpose([2, 0, 1])
    return img

#img [h, w, c]
def Generate_PadRow(img):
    # img = img.transpose([1, 2, 0])
    h = img.shape[0]
    w = img.shape[1]
    # h_p = random.randint(0, 16)
    # h_p = random.randint(0, 30)
    h_p = random.randint(0, 24)
    y0 = int(random.random() * h)
    if y0 + h_p < h:
        col_s = y0
        col_e = y0 + h_p
    else:
        col_s = y0 - h_p
        col_e = y0
    img[col_s:col_e, 0:w, :] = [0, 0, 0]
    # img = img.transpose([2, 0, 1])
    return img

#img [h, w, c]
def local_medblur(img):
    #img = img.transpose([1, 2, 0])
    ksize = random.randint(0, 1) * 2 + 7
    h = img.shape[0]
    w = img.shape[1]
    wsize = int(random.randint(20, 50) * 1.0 / 100.0 * w)
    x0 = min(int(random.random() * w), w - wsize + 5)
    y0 = 0
    dst = cv2.medianBlur(img[:, x0:x0 + wsize, :], ksize)
    img[:, x0:x0 + wsize, :] = dst
    #img = img.transpose([2, 0, 1])
    return img

#img [h, w, c]
def motion_blur(img):
    # img = img.transpose([1, 2, 0])
    # h = img.shape[0]
    # w = img.shape[1]
    # wsize = int(random.randint(20, 50) * 1.0 / 100.0 * w)
    # x0 = min(int(random.random() * w), w - wsize + 5)
    # y0 = 0
    image = numpy.array(img)
    degree = random.randint(8,12)
    angle = random.randint(45,60)
    # degree = 16
    # angle = 60
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = numpy.diag(numpy.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0.0, 255.0, cv2.NORM_MINMAX)
    # img = blurred.transpose([2, 0, 1])
    img = blurred
    return img

#img [h, w, c]
def gaussian_blur(img):
    # img = img.transpose([1, 2, 0])
    h = img.shape[0]
    w = img.shape[1]
    wsize = int(random.randint(20, 50) * 1.0 / 100.0 * w)
    x0 = min(int(random.random() * w), w - wsize + 5)
    y0 = 0
    knsize = random.randint(0, 3) * 2 + 9
    blurred = cv2.GaussianBlur(img[:,x0:x0 + wsize,:], ksize=(knsize, knsize), sigmaX=0, sigmaY=0)
    img[:,x0:x0 + wsize,:] = blurred
    # img = img.transpose([2, 0, 1])
    return img

def random_scale_downup(img, size_ratio_down=(0.4, 0.75)):
    random_ratio = random.uniform(size_ratio_down[0], size_ratio_down[1])
    h, w, c = img.shape
    h_down = int(random_ratio * h + 0.5)
    w_down = int(random_ratio * w + 0.5)

    img = cv2.resize(img, (w_down, h_down))
    img = cv2.resize(img, (w, h))
    return img