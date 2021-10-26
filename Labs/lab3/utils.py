import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters

from scipy.ndimage import affine_transform, rotate

# Functions to convert points to homogeneous coordinates and back
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]

def cv2_imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches,
                 keypoints_color='k', matches_color=None, only_matches=False):
    """Plot matched features.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    """

    image1.astype(np.float32)
    image2.astype(np.float32)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    image = np.concatenate([image1, image2], axis=1)

    offset = image1.shape

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0],
                   facecolors='none', edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset[1], keypoints2[:, 0],
                   facecolors='none', edgecolors=keypoints_color)

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, 2 * offset[1], offset[0], 0))

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]
        # print ("U ", idx1, idx2, keypoints1[idx1], keypoints2[idx2])
        if matches_color is None:
            color = np.random.rand(3)
        else:
            color = matches_color

        ax.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
                (keypoints1[idx1, 0], keypoints2[idx2, 0]),
'-', color=color)


def flip_keypoints(keypoints, im_shape, xy=None):
    xy = 0 if xy == 0 else 1
    copied_keypoints = keypoints.copy()
    for kp in copied_keypoints:
        kp[xy] = im_shape[xy]-1-kp[xy]
    return copied_keypoints
        
def plot_mirror_matches(ax, im, kps, matches):
    plot_matches(ax, im, np.fliplr(im), kps, flip_keypoints(kps, im.shape), matches)

def plot_self_matches(ax, im, kps, matches):
    plot_matches(ax, im, im, kps, kps, matches)
        
def warp_image(src, dst, h_matrix):
    """
    Warps the destination image using the homography matrix and concatenate them together.
    """
    dst = cv2.transpose(dst).copy()
    src = cv2.transpose(src).copy()
    dst = cv2.warpPerspective(dst, np.linalg.inv(h_matrix), (src.shape[1], src.shape[0] + dst.shape[0]))
    dst[0:src.shape[0], 0:src.shape[1]] = src
    dst = cv2.transpose(dst)
    
    return dst
        
def draw_mirror_line(im, rs, thetas):
    for r, theta in zip(rs, thetas):
        if np.pi / 4 < theta < 3 * (np.pi / 4):
            for x in range(im.shape[1]):
                try:
                    y = int((r - x * np.cos(theta)) / np.sin(theta))
                    im[y-2][x][2] = 255
                    im[y-1][x][2] = 255
                    im[y][x][2] = 255
                    im[y+1][x][2] = 255
                    im[y+2][x][2] = 255

                except:
                    continue
        else:
            for y in range(im.shape[0]):
                try:
                    x = int((r-y*np.sin(theta))/np.cos(theta))
                    im[y][x-2][2] = 255
                    im[y][x-1][2] = 255
                    im[y][x][2] = 255
                    im[y][x+1][2] = 255 
                    im[y][x+2][2] = 255
                except IndexError:
                    continue

    # draw plot 
    plt.figure(figsize=(7,7))
    plt.imshow(im)
    plt.axis('off') 
    plt.show()
    
def draw_centers(img, Y, X):

    im = img.copy()
    h,w = im.shape[:2]
    
    for x,y in zip(X,Y):
        
        im[y-2:y+2,x-50:x+50,0] = 0
        im[y-2:y+2,x-50:x+50,1] = 255
        im[y-2:y+2,x-50:x+50,2] = 0
        im[y-50:y+50,x-2:x+2,0] = 0
        im[y-50:y+50,x-2:x+2,1] = 255
        im[y-50:y+50,x-2:x+2,2] = 0
    plt.imshow(im)