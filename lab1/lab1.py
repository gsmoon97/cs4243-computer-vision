import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: image preprossessing #####


def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return

    ###Your code here###
    # extract each channel from the source image
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    # calculate the sum of respectively weighted color channels
    img_gray = r_channel * 0.299 + g_channel * 0.587 + b_channel * 0.114
    ###
    return img_gray


def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=float)
    sobelv = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype=float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype=float)

    ###Your code here####
    # initialize the output images
    height, width = img.shape
    img_grad_h = np.zeros((height, width))
    img_grad_v = np.zeros((height, width))
    img_grad_d1 = np.zeros((height, width))
    img_grad_d2 = np.zeros((height, width))
    # convert source image to float dtype
    # add extra layer of padding to source image
    img_pad = np.zeros((height + 2 * 1, width + 2 * 1))
    img_pad[1: -1, 1: -1] = img[:, :].astype(float)
    # flip the gradient filters before operation
    flipped_sobelh = np.flipud(np.fliplr(sobelh))
    flipped_sobelv = np.flipud(np.fliplr(sobelv))
    flipped_sobeld1 = np.flipud(np.fliplr(sobeld1))
    flipped_sobeld2 = np.flipud(np.fliplr(sobeld2))
    # iterate over each pixel to compute the gradient components
    for ri in range(height):
        for ci in range(width):
            img_grad_h[ri][ci] = (
                img_pad[ri:ri + 3, ci:ci + 3] * flipped_sobelh
            ).sum()
    for ri in range(height):
        for ci in range(width):
            img_grad_v[ri][ci] = (
                img_pad[ri:ri + 3, ci:ci + 3] * flipped_sobelv
            ).sum()
    for ri in range(height):
        for ci in range(width):
            img_grad_d1[ri][ci] = (
                img_pad[ri:ri + 3, ci:ci + 3] * flipped_sobeld1
            ).sum()
    for ri in range(height):
        for ci in range(width):
            img_grad_d2[ri][ci] = (
                img_pad[ri:ri + 3, ci:ci + 3] * flipped_sobeld2
            ).sum()
    ###
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2


def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img.
    """
    height, width = img.shape[:2]
    new_height, new_width = (
        height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(
        img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    ###Your code here###
    # convert img_pad to source image dtype
    img_pad = img_pad.astype(img.dtype)
    # apply the source image onto the output image
    img_pad[pad_height_bef: - pad_height_aft,
            pad_width_bef: - pad_width_aft] = img[:, :]
    ###
    return img_pad


##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 6 nested for-loops.
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    response = np.zeros((Ho, Wo))
    # iterate over each pixel to compute the gradient components
    for ri in range(Ho):
        for ci in range(Wo):
            for rt in range(Hk):
                for ct in range(Wk):
                    response[ri][ci] += (img[ri + rt][ci + ct].astype(float)
                                         * template[rt][ct]).sum()
            response[ri][ci] /= (np.linalg.norm(template)
                                 * np.linalg.norm(img[ri:ri + Hk, ci:ci + Wk].astype(float)))
    ###
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops.
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    # initialize the output image
    response = np.zeros((Ho, Wo))
    # iterate over each pixel to compute the gradient components
    for ri in range(Ho):
        for ci in range(Wo):
            window = img[ri:ri + Hk, ci:ci + Wk].astype(float)
            response[ri][ci] = (window * template).sum() / (
                np.linalg.norm(template) * np.linalg.norm(window))
    ###
    return response


def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    # reshape image and template for matrix multiplication
    Pr = np.zeros((Ho * Wo, 3 * Hk * Wk))
    Kr = template.transpose(2, 0, 1).reshape(-1, 1)
    Ki = np.full(3 * Hk * Wk, 1, dtype=float).reshape(-1, 1)
    img_t = img.transpose(2, 0, 1)
    ro = 0
    for r in range(Ho):
        for c in range(Wo):
            red = img_t[0][r:r+Hk, c:c+Wk].reshape(-1)
            green = img_t[1][r:r+Hk, c:c+Wk].reshape(-1)
            blue = img_t[2][r:r+Hk, c:c+Wk].reshape(-1)
            Pr[ro] = np.concatenate([red, green, blue])
            ro += 1
    Psqr = np.square(Pr)
    Mw = np.sqrt(np.dot(Psqr, Ki))

    response = (np.dot(Pr, Kr)
                / (Mw * np.linalg.norm(template))
                ).reshape(Ho, Wo)
    ###
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
        1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
        3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range).
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    ###Your code here###
    if threshold is None:
        threshold = 0.85

    Hr, Wr = response.shape[:2]
    Hs, Ws = suppress_range

    # 1. Set X < threshold to 0
    X = np.where(response < threshold, 0, response)
    res = np.zeros((Hr, Wr), dtype=float)

    # 2. While there are non-zero values in X
    global_max = np.max(X)
    while global_max != 0:
        # a. Find the global maximum in X and record the coordinates as a local maximum.
        max_rows, max_cols = np.where(global_max == X)

        row = max_rows[0]
        col = max_cols[0]

        max_rows = np.delete(max_rows, 0)
        max_cols = np.delete(max_cols, 0)

        res[row, col] = 1

        # b. Set a small window of size w×w points centered on the found maximum to 0.
        row_bottom = max(0, row - Hs)
        row_top = min(Hr - 1, row + Hs)
        col_left = max(0, col - Ws)
        col_right = min(Wr - 1, col + Ws)
        X[row_bottom: row_top +
            1, col_left: col_right + 1] = 0

        global_max = np.max(X)
    ###
    return res

##### Part 4: Question And Answer #####


def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    response = np.zeros((Ho, Wo))
    # iterate over each pixel to compute the gradient components
    template_t = template.copy().astype(float).transpose(2, 0, 1)
    for j in range(3):
        template_t[j] = template_t[j] - template_t[j].mean()
    ms_template = template_t.transpose(1, 2, 0)
    for ri in range(Ho):
        for ci in range(Wo):
            window = img[ri:ri + Hk, ci:ci + Wk].astype(float)
            window_t = window.transpose(2, 0, 1)
            for i in range(3):
                window_t[i] = window_t[i] - window_t[i].mean()
            window = window_t.transpose(1, 2, 0)
            response[ri][ci] = (window * ms_template).sum() / (
                np.linalg.norm(window) * np.linalg.norm(ms_template))
    ###
    return response


###############################################
"""Helper functions: You should not have to touch the following functions.
"""


def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15, 15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(
                imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)


def show_img_with_squares(response, img_ori=None, rec_shape=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()
    H, W = response.shape[:2]
    if rec_shape is None:
        h_rec, w_rec = 25, 25
    else:
        h_rec, w_rec = rec_shape

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.rectangle(
            response, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (255, 0, 0), 2)
        if img_ori is not None:
            img_ori = cv2.rectangle(
                img_ori, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (0, 255, 0), 2)

    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)
