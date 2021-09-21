from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


##################### TASK 1 ###################

# 1.1 IMPLEMENT
def make_gaussian_kernel(ksize, sigma):
    '''
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    # YOUR CODE HERE
    min_x = min_y = - (ksize // 2)  # -3 for 6 and 7
    max_x = max_y = - (-ksize // 2) - 1  # 2 for 6 # 3 for 7
    x_coordinates, y_coordinates = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
    kernel = np.exp(((x_coordinates - np.mean(x_coordinates)) ** 2 + (y_coordinates - np.mean(y_coordinates)) ** 2) /
                    (-2.0 * (sigma ** 2)))
    # END

    return kernel / kernel.sum()

# GIVEN


def cs4243_filter(image, kernel):
    """
    Fast version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk), also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and reshape back
    to get the final output image.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    def cs4243_rotate180(kernel):
        kernel = np.flip(np.flip(kernel, 0), 1)
        return kernel

    def img2col(input, h_out, w_out, h_k, w_k, stride):
        h, w = input.shape
        out = np.zeros((h_out*w_out, h_k*w_k))

        convwIdx = 0
        convhIdx = 0
        for k in range(h_out*w_out):
            if convwIdx + w_k > w:
                convwIdx = 0
                convhIdx += stride
            out[k] = input[convhIdx:convhIdx+h_k,
                           convwIdx: convwIdx+w_k].flatten()
            convwIdx += stride
        return out

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    if Hk % 2 == 0 or Wk % 2 == 0:
        raise ValueError

    hkmid = Hk//2
    wkmid = Wk//2

    image = cv2.copyMakeBorder(
        image, hkmid, hkmid, wkmid, wkmid, cv2.BORDER_REFLECT)
    filtered_image = np.zeros((Hi, Wi))
    kernel = cs4243_rotate180(kernel)
    col = img2col(image, Hi, Wi, Hk, Wk, 1)
    kernel_flatten = kernel.reshape(Hk*Wk, 1)
    output = col @ kernel_flatten
    filtered_image = output.reshape(Hi, Wi)

    return filtered_image

# GIVEN


def cs4243_blur(img, gaussian_kernel, display=True):
    '''
    Performing Gaussian blurring on an image using a Gaussian kernel.
    :param img: input image
    :param gaussian_kernel: gaussian kernel
    :return blurred_img: blurred image
    '''

    blurred_img = cs4243_filter(img, gaussian_kernel)

    if display:

        fig1, axes_array = plt.subplots(1, 2)
        fig1.set_size_inches(8, 4)
        image_plot = axes_array[0].imshow(img, cmap=plt.cm.gray)
        axes_array[0].axis('off')
        axes_array[0].set(title='Original Image')
        image_plot = axes_array[1].imshow(blurred_img, cmap=plt.cm.gray)
        axes_array[1].axis('off')
        axes_array[1].set(title='Filtered Image')
        plt.show()
    return blurred_img

# 2 IMPLEMENT


def estimate_gradients(original_img, display=True):
    '''
    Compute gradient orientation and magnitude for the input image.
    Perform the following steps:

    1. Compute dx and dy, responses to the vertical and horizontal Sobel kernel. Make use of the cs4243_filter function.

    2. Compute the gradient magnitude which is equal to sqrt(dx^2 + dy^2) 

    3. Compute the gradient orientation using the following formula:
        gradient = atan2(dy/dx)

    You may want to divide the original image pixel value by 255 to prevent overflow.

    Note that our axis choice is as follows:
            --> y
            |   
            ↓ x    
    Where img[x,y] denotes point on the image at coordinate (x,y)

    :param original_img: original grayscale image
    :return d_mag: gradient magnitudes matrix
    :return d_angle: gradient orientation matrix (in radian)
    '''

    dx = None
    dy = None
    d_mag = None
    d_angle = None

    # YOUR CODE HERE
    '''
    HINT:
    In the lecture, 
    
    Sx =  1  0 -1
          2  0 -2
          1  0 -1
          
    Sy =  1  2  1
          0  0  0
         -1 -2 -1
         
    Here:
    
    Kx = [[ 1,  2,  1],
          [ 0,  0,  0],
          [-1, -2, -1]]
    
    Ky = [[ 1,  0, -1],
          [ 2,  0, -2],
          [ 1,  0, -1]]
 
    This is because x direction is the downward line.
    '''
    sobel_h = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=float)
    sobel_v = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=float)
    # divide intensity value of each pixel by 255 to prevent overflow
    normalized_img = original_img / 255
    dx = cs4243_filter(normalized_img, sobel_v)
    dy = cs4243_filter(normalized_img, sobel_h)
    d_mag = np.sqrt(dx ** 2 + dy ** 2)
    d_angle = np.arctan2(dx, dy)
    # END
    if display:

        fig2, axes_array = plt.subplots(1, 4)
        fig2.set_size_inches(16, 4)
        image_plot = axes_array[0].imshow(d_mag, cmap='gray')
        axes_array[0].axis('off')
        axes_array[0].set(title='Gradient Magnitude')

        image_plot = axes_array[1].imshow(dx, cmap='gray')
        axes_array[1].axis('off')
        axes_array[1].set(title='dX')

        image_plot = axes_array[2].imshow(dy, cmap='gray')
        axes_array[2].axis('off')
        axes_array[2].set(title='dY')

        image_plot = axes_array[3].imshow(d_angle, cmap='gray')
        axes_array[3].axis('off')
        axes_array[3].set(title='Gradient Direction')
        plt.show()

    return d_mag, d_angle

# 3a IMPLEMENT


def non_maximum_suppression(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix without interpolation.
    Split the range -180° ~ 180° into 8 even ranges. For each pixel, determine which range the gradient
    orientation belongs to and pick the corresponding two pixels from the adjacent eight pixels surrounding 
    that pixel. Keep the pixel if its value is larger than the other two.
    Do note that the coordinate system is as below and angular measurements are clockwise.
    ----------→ y  
    |
    |
    |
    |        x X x
    ↓ x       \|/   
             x-o-x  
              /|\    
             x X x 
         -22.5 0 22.5

    For instance, in the example above if the orientation at the coordinate of interest (x,y) is 20°, it belongs to the -22.5°~22.5° range, and the two pixels to be compared with are at (x+1,y) and (x-1,y) (aka the two big X's). If the angle was instead 40°, it belongs to the 22.5°-67.5° and the two pixels we need to consider will be (x+1, y+1) and (x-1,y-1)

    There are only 4 sets of offsets: (0,1), (1,0), (1,1), and (1,-1), since to find the second pixel offset you just need 
    to multiply the first tuple by -1.

    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    # Change angles to degrees to improve quality of life
    d_angle_180 = d_angle * 180/np.pi

    # YOUR CODE HERE
    height, width = d_angle_180.shape
    # ignore pixels in image borders
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if -22.5 <= d_angle_180[i][j] < 22.5 or d_angle_180[i][j] < -157.5 or 157.5 <= d_angle_180[i][j]:
                offset = (0, 1)
            elif 22.5 <= d_angle_180[i][j] < 67.5 or -157.5 <= d_angle_180[i][j] < -112.5:
                offset = (1, 1)
            elif 67.5 <= d_angle_180[i][j] < 112.5 or -112.5 <= d_angle_180[i][j] < -67.5:
                offset = (1, 0)
            elif 112.5 <= d_angle_180[i][j] < 157.5 or -67.5 <= d_angle_180[i][j] < 22.5:
                offset = (1, -1)
            if d_mag[i][j] > d_mag[i + offset[0]][j + offset[1]] and d_mag[i][j] > d_mag[i - offset[0]][j - offset[1]]:
                out[i][j] = d_mag[i][j]
    # END
    if display:
        _ = plt.figure(figsize=(10, 10))
        plt.imshow(out)
        plt.title("Suppressed image (without interpolation)")

    return out

# 3b IMPLEMENT


def non_maximum_suppression_interpol(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix with interpolation.
    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    d_angle_180 = d_angle * 180/np.pi

    # YOUR CODE HERE
    height, width = d_angle_180.shape
    # ignore pixels in image borders
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if 0.0 <= d_angle_180[i][j] < 45.0 or d_angle_180[i][j] < -135.0:
                offset_1 = (0, 1)
                offset_2 = (1, 1)
            elif 45.0 <= d_angle_180[i][j] < 90.0 or -135.0 <= d_angle_180[i][j] < -90.0:
                offset_1 = (1, 1)
                offset_2 = (1, 0)
            elif 90.0 <= d_angle_180[i][j] < 135.0 or -90.0 <= d_angle_180[i][j] < -45.0:
                offset_1 = (1, 0)
                offset_2 = (1, -1)
            elif 135.0 <= d_angle_180[i][j] or -45.0 <= d_angle_180[i][j] < 0.0:
                offset_1 = (1, -1)
                offset_2 = (1, 1)
            value_a = math.tan(d_angle[i][j]) * (d_mag[i + offset_2[0]][j + offset_2[1]]
                                                 - d_mag[i + offset_1[0]][j + offset_2[1]]) + d_mag[i + offset_1[0]][j + offset_2[1]]
            value_b = math.tan(d_angle[i][j]) * (d_mag[i - offset_2[0]][j - offset_2[1]]
                                                 - d_mag[i - offset_1[0]][j - offset_2[1]]) + d_mag[i - offset_1[0]][j - offset_2[1]]
            if d_mag[i][j] > value_a and d_mag[i][j] > value_b:
                out[i][j] = d_mag[i][j]
    # END
    if display:
        _ = plt.figure(figsize=(10, 10))
        plt.imshow(out, cmap='gray')
        plt.title("Suppressed image (with interpolation)")

    return out

# 4 IMPLEMENT


def double_thresholding(inp, perc_weak=0.1, perc_strong=0.3, display=True):
    '''
    Perform double thresholding. Use on the output of NMS. The high and low thresholds are computed as follow:

    delta = max_val - min_val
    high_threshold = min_val + perc_strong * delta 
    low_threshold = min_val + perc_weak * delta

    perc_weak being 0 is possible
    Do note that the return edge images should be binary (0-1 or True-False)
    :param inp: numpy.ndarray
    :param perc_weak: value to determine low threshold
    :param perc_strong: value to determine high threshold
    :return weak_edges, strong_edges: binary edge images
    '''
    weak_edges = strong_edges = None

    # YOUR CODE HERE

    # END

    if display:

        fig2, axes_array = plt.subplots(1, 2)
        fig2.set_size_inches(10, 5)
        image_plot = axes_array[0].imshow(strong_edges, cmap='gray')
        axes_array[0].axis('off')
        axes_array[0].set(title='Strong ')

        image_plot = axes_array[1].imshow(weak_edges, cmap='gray')
        axes_array[1].axis('off')
        axes_array[1].set(title='Weak')

    return weak_edges, strong_edges

# 5 IMPLEMENT


def edge_linking(weak, strong, n=200, display=True):
    '''
    Perform edge-linking on two binary weak and strong edge images. 
    A weak edge pixel is linked if any of its eight surrounding pixels is a strong edge pixel.
    You may want to avoid using loops directly due to the high computational cost. One possible trick is to generate
    8 2D arrays from the strong edge image by offseting and sum them together; entries larger than 0 mean that at least one surrounding
    pixel is a strong edge pixel (otherwise the sum would be 0).

    You may also want to limit the number of iterations (test with 10-20 iterations first to check your implementation speed), and use a stopping condition (stop if no more pixel is added to the strong edge image).
    Also, when a weak edge pixel is added to the strong set, remember to remove it.


    :param weak: weak edge image (binary)
    :param strong: strong edge image (binary)
    :param n: maximum number of iterations
    :return out: final edge image
    '''
    assert weak.shape == strong.shape, \
        "Weak and strong edge image have to have the same dimension"
    out = None

    # YOUR CODE HERE

    # END
    if display:
        _ = plt.figure(figsize=(10, 10))
        plt.imshow(s)
        plt.title("Edge image")
    return out

##################### TASK 2 ######################

# 1/2/3 IMPLEMENT


def hough_vote_lines(img):
    '''
    Use the edge image to vote for 2 parameters: distance and theta
    Beware of our coordinate convention.

    You may find the np.linspace function useful.

    :param img: edge image
    :return A: accumulator array
    :return distances: distance values array
    :return thetas: theta values array
    '''
    # YOUR CODE HERE

    # END

    return A, distances, thetas


# 4 GIVEN


def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters).

    Also include the array of values corresponding to the bins, in descending order.

    e.g.
    Suppose for a line detection case, you get the following output:
    [
    [122, 101, 93],
    [3,   40,  21],
    [0,   1.603, 1.605]
    ]
    This means that the local maxima with the highest vote gets a vote score of 122, and the corresponding parameter value is distance=3, 
    theta = 0.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(
    ), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])]
                           for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res


##################### TASK 3 ######################

# 1/2/3 IMPLEMENT


def hough_vote_circles(img, radius=None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle]. This parameter is very useful for your experimentation later on (e.g. if there are only large circles then you don't have to keep R_min very small).

    Hint: You can use the function circle_perimeter to make a circular mask. Center the mask over the accumulator array and increment the array. In this case, you will have to pad the accumulator array first, and clip it afterwards. 
    Remember that the return accumulator array should have matching dimension with the lengths of the parameter ranges. 

    The dimensions of the accumulator array should be in this order: radius, x-coordinate, y-coordinate.

    :param img: edge image
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''

    # Check the radius range
    h, w = img.shape[:2]
    if radius == None:
        R_max = np.hypot(h, w)
        R_min = 3
    else:
        [R_min, R_max] = radius

    # YOUR CODE HERE

    # END

    return A, R, X, Y

##################### TASK 4 ######################

# IMPLEMENT


def hough_vote_circles_grad(img, d_angle, radius=None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle].
    This time, gradient information is used to avoid casting too many unnecessary votes.

    Remember that for a given pixel, you need to cast two votes along the orientation line. One in the positive direction, the other in
    negative direction.

    :param img: edge image
    :param d_angle: corresponding gradient orientation matrix
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''
    # Check the radius range
    h, w = img.shape[:2]
    if radius == None:
        R_max = np.hypot(h, w)
        R_min = 3
    else:
        [R_min, R_max] = radius

    # YOUR CODE HERE

    # END
    return A, R, X, Y


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


def draw_lines(hspace, dists, thetas, hs_maxima, file_path):
    im_c = read_img(file_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(im_c, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(thetas).mean()
    d_step = 0.5 * np.diff(dists).mean()
    bounds = [np.rad2deg(thetas[0] - angle_step),
              np.rad2deg(thetas[-1] + angle_step),
              dists[-1] + d_step, dists[0] - d_step]

    ax[1].imshow(np.log(1 + hspace), extent=bounds,
                 cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(im_c, cmap=cm.gray)
    ax[2].set_ylim((im_c.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    # You may want to change the codes below if you use a different axis choice.
    for _, dist, angle in zip(*hs_maxima):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((y0, x0), slope=np.tan(np.pi-angle))

    plt.tight_layout()
    plt.show()


def draw_circles(local_maxima, file_path, title):
    img = cv2.imread(file_path)
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    circle = []
    for _, r, x, y in zip(*local_maxima):
        circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.title(title)
    plt.show()
