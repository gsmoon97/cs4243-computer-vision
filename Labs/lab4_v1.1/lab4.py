import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

# Part 1


def detect_points(img, min_distance, rou, pt_num, patch_size, tau_rou, gamma_rou):
    """
    Patchwise Shi-Tomasi point extraction.

    Hints:
    (1) You may find the function cv2.goodFeaturesToTrack helpful. The initial default parameter setting is given in the notebook.

    Args:
        img: Input RGB image.
        min_distance: Minimum possible Euclidean distance between the returned corners. A parameter of cv2.goodFeaturesToTrack
        rou: Parameter characterizing the minimal accepted quality of image corners. A parameter of cv2.goodFeaturesToTrack
        pt_num: Maximum number of corners to return. A parameter of cv2.goodFeaturesToTrack
        patch_size: Size of each patch. The image is divided into several patches of shape (patch_size, patch_size). There are ((h / patch_size) * (w / patch_size)) patches in total given a image of (h x w)
        tau_rou: If rou falls below this threshold, stops keypoint detection for that patch
        gamma_rou: Decay rou by a factor of gamma_rou to detect more points.
    Returns:
        pts: Detected points of shape (N, 2), where N is the number of detected points. Each point is saved as the order of (height-corrdinate, width-corrdinate)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, c = img.shape

    # The required number of keypoints for each patch. `pt_num` is used as a parameter, while `Np` is used as a stopping criterion.
    Np = pt_num * 0.9

    # YOUR CODE HERE
    pts = np.zeros((1, 1, 2), dtype='float64')
    h_bins = h // patch_size
    w_bins = w // patch_size
    for h_idx in range(h_bins):
        for w_idx in range(w_bins):
            patch_mask = np.zeros((h, w), dtype='uint8')
            patch_mask[h_idx * patch_size:(h_idx+1) * patch_size,
                       w_idx * patch_size:(w_idx+1) * patch_size] = 1
            patch_pts = cv2.goodFeaturesToTrack(
                img_gray, pt_num, rou, min_distance, mask=patch_mask)
            patch_rou = rou
            while (patch_pts is None or len(patch_pts) < Np) and patch_rou >= tau_rou:
                patch_rou *= gamma_rou
                patch_pts = cv2.goodFeaturesToTrack(
                    img_gray, pt_num, patch_rou, min_distance, mask=patch_mask)
            if patch_pts is not None:
                pts = np.concatenate((pts, patch_pts), axis=0)
    pts = np.squeeze(pts)
    # goodFeaturestoTrack returns (x, y) coordinates
    # need to swap the order to return (y, x) coordinates
    for p in pts:
        p[0], p[1] = p[1], p[0]
    # END

    return pts


def extract_point_features(img, pts, window_patch):
    """
    Extract patch feature for each point.

    The patch feature for a point is defined as the patch extracted with this point as the center.

    Note that the returned pts is a subset of the input pts.
    We discard some of the points as they are close to the boundary of the image and we cannot extract a full patch.

    Please normalize the patch features by subtracting the mean intensity and dividing by its standard deviation.

    Args:
        img: Input RGB image.
        pts: Detected point corners from detect_points().
        window_patch: The window size of patch cropped around the point. The final patch is of size (5 + 1 + 5, 5 + 1 + 5) = (11, 11). The center is the given point.
                      For example, suppose an image is of size (300, 400). The point is located at (50, 60). The window size is 5.
                      Then, we use the cropped patch, i.e., img[50-5:50+5+1, 60-5:60+5+1], as the feature for that point. The patch size is (11, 11), so the dimension is 11x11=121.
    Returns:
        pts: A subset of the input points. We can extract a full patch for each of these points.
        features: Patch features of the points of the shape (N, (window_patch*2 + 1)^2), where N is the number of points
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE
    # discard points that are close to the image boundaries
    h_min = window_patch
    h_max = h - window_patch - 1
    w_min = window_patch
    w_max = w - window_patch - 1
    within_border = np.apply_along_axis(
        lambda p: h_min <= p[0] <= h_max and w_min <= p[1] <= w_max, 1, pts)
    pts = pts[within_border, :]
    features = np.zeros((1, (window_patch * 2 + 1)**2), dtype='float64')
    for p in pts:
        y, x = p.astype(int)
        patch = img_gray[y - window_patch: y + window_patch + 1,
                         x - window_patch: x + window_patch + 1]
        # normalize the patch
        normalized_patch = (patch - np.mean(patch)) / np.std(patch)
        features = np.concatenate(
            (features, [normalized_patch.flatten()]), axis=0)
    # End

    return pts, features


def mean_shift_clustering(features, bandwidth):
    """
    Mean-Shift Clustering.

    There are various ways of implementing mean-shift clustering.
    The provided default bandwidth value may not be optimal to your implementation.
    Please fine-tune the bandwidth so that it can give the best result.

    Args:
        img: Input RGB image.
        bandwidth: If the distance between a point and a clustering mean is below bandwidth, this point probably belongs to this cluster.
    Returns:
        clustering: A dictionary, which contains three keys as follows:
                    1. cluster_centers_: a numpy ndarrary of shape [N_c, 2] for the naive point cloud task and [N_c, 121] for the main task (patch features of the point).
                                         Each row is the center of that cluster.
                    2. labels_:  a numpy nadarray of shape [N,], where N is the number of features.
                                 labels_[i] denotes the label of the i-th feature. The label is between [0, N_c - 1]
                    3. bandwidth: bandwith value
    """
    # YOUR CODE HERE
    # ===============================================================
    # no_of_features = features.shape[0]
    # centroids = features.copy()
    # while True:
    #     new_centroids = []
    #     for centroid in centroids:
    #         within_bandwidth = []
    #         for feature in features:
    #             if np.linalg.norm(feature - centroid) < bandwidth:
    #                 within_bandwidth.append(feature)
    #         # compute new centroid based on the window
    #         new_centroid = np.mean(within_bandwidth, axis=0)
    #         new_centroids.append(new_centroid)
    #     # remove duplicate centroids
    #     new_centroids = np.array(np.unique(new_centroids, axis=0))
    #     if np.array_equal(centroids, new_centroids):
    #         break
    #     else:
    #         centroids = new_centroids
    # labels = []
    # no_of_centroids = centroids.shape[0]
    # for feature in features:
    #     min_dist = math.inf
    #     min_idx = -1
    #     for c_idx in range(no_of_centroids):
    #         dist = np.linalg.norm(feature - centroids[c_idx])
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_idx = c_idx
    #     labels.append(min_idx)
    # labels = np.array(labels)
    # ===============================================================
    no_of_features = features.shape[0]
    centroids = {}
    for i in range(no_of_features):
        centroids[i] = features[i]
    while True:
        converged = True
        for f_idx, centroid in centroids.items():
            within_bandwidth = []
            for feature in features:
                if np.linalg.norm(feature - centroid) < bandwidth:
                    within_bandwidth.append(feature)
            new_centroid = np.mean(within_bandwidth, axis=0)
            # if np.linalg.norm(centroid - new_centroid) > 0:
            if not np.array_equal(centroid, new_centroid):
                converged = False
                centroids.update({f_idx: new_centroid})
        if converged:
            break
    unique_centroids = np.unique(np.array(list(centroids.values())), axis=0)
    labels = []
    for centroid in centroids.values():
        idx = np.where(unique_centroids == centroid)[0][0]
        labels.append(idx)
    labels = np.array(labels)
    clustering = {'cluster_centers_': unique_centroids,
                  'labels_': labels, 'bandwidth': bandwidth}
    # END

    return clustering


def cluster(img, pts, features, bandwidth, tau1, tau2, gamma_h):
    """
    Group points with similar appearance, then refine the groups.

    "gamma_h" provides another way of fine-tuning bandwidth to avoid the number of clusters becoming too large.
    Alternatively, you can ignore "gamma_h" and fine-tune bandwidth by yourself.

    Args:
        img: Input RGB image.
        pts: Output from `extract_point_features`.
        features: Patch feature of points. Output from `extract_point_features`.
        bandwidth: Window size of the mean-shift clustering. In pdf, the bandwidth is represented as "h", but we use "bandwidth" to avoid the confusion with the image height
        tau1: Discard clusters with less than tau1 points
        tau2: Perform further clustering for clusters with more than tau2 points using K-means
        gamma_h: To avoid the number of clusters becoming too large, tune the bandwidth by gradually increasing the bandwidth by a factor gamma_h
    Returns:
        clusters: A list of clusters. Each cluster is a numpy ndarray of shape [N_cp, 2]. N_cp is the number of points of that cluster.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE
    mean_shift_clusters = mean_shift_clustering(features, bandwidth)
    cluster_centers = mean_shift_clusters['cluster_centers_']
    labels = mean_shift_clusters['labels_']
    no_of_clusters = cluster_centers.shape[0]
    no_of_features = features.shape[0]
    candidate_clusters = [[] for _ in range(no_of_clusters)]
    clusters = []
    for i in range(no_of_features):
        candidate_clusters[labels[i]].append(features[i])
    for cluster in candidate_clusters:
        cluster_size = len(cluster)
        if cluster_size < tau1:  # discard small clusters
            print('detected small cluster')
            continue
        elif cluster_size > tau2:  # partition large clusters
            print('detected big cluster')
            kmeans = KMeans(n_clusters=(cluster_size // tau2)).fit(cluster)
            kmeans_cluster_centers = kmeans.cluster_centers_
            kmeans_labels = kmeans.labels_
            no_of_kmeans_clusters = kmeans_cluster_centers.shape[0]
            kmeans_clusters = [[] for _ in range(no_of_kmeans_clusters)]
            for idx, point in enumerate(cluster):
                kmeans_clusters[labels[idx]].append(point)
            for kmeans_cluster in kmeans_clusters:
                clusters.append(np.array(kmeans_cluster))
            continue
        else:
            print('detected new cluster')
            clusters.append(np.array(cluster))
    # END

    return clusters

# Part 2


def get_proposal(pts_cluster, tau_a, X):
    """
    Get the lattice proposal

    Hints:
    (1) As stated in the lab4.pdf, we give priority to points close to each other when we sample a triplet.
        This statement means that we can start from the three closest points and iterate N_a times.
        There is no need to go through every triplet combination.
        For instance, you can iterate over each point. For each point, you choose 2 of the 10 nearest points. The 3 points form a triplet.
        In this case N_a = num_points * 45.

    (2) It is recommended that you reorder the 3 points.
        Since {a, b, c} are transformed onto {(0, 0), (1, 0), (0, 1)} respectively, the point a is expected to be the vertex opposite the longest side of the triangle formed by these three points

    (3) Another way of refining the choice of triplet is to keep the triplet whose angle (between the edges <a, b> and <a, c>) is within a certain range.
        The range, for instance, is between 20 degrees and 120 degrees.

    (4) You may find `cv2.getAffineTransform` helpful. However, be careful about the HW and WH ordering when you use this function.

    (5) If two triplets yield the same number of inliers, keep the one with closest 3 points.

    Args:
        pts_cluster: Points within the same cluster.
        tau_a: The threshold of the difference between the transformed corrdinate and integer positions.
               For example, if a point is transformed into (1.1, -2.03), the closest integer position is (1, -2), then the distance is sqrt(0.1^2 + 0.03^2) (for Euclidean distance case).
               If it is smaller than "tau_a", we consider this point as inlier.
        X: When we compute the inliers, we only consider X nearest points to the point "a".
    Returns:
        proposal: A list of inliers. The first 3 inliers are {a, b, c}.
                  Each inlier is a dictionary, with key of "pt_int" and "pt" representing the integer positions after affine transformation and orignal coordinates.
    """
    # YOU CODE HERE

    # END

    return proposal


def find_texels(img, proposal, texel_size=50):
    """
    Find texels from the given image.

    Hints:
    (1) This function works on RGB image, unlike previous functions such as point detection and clustering that operate on grayscale image.

    (2) You may find `cv2.getPerspectiveTransform` and `cv2.warpPerspective` helpful.
        Please refer to the demo in the notebook for the usage of the 2 functions.
        Be careful about the HW and WH ordering when you use this function.

    (3) As stated in the pdf, each texel is defined by 3 or 4 inlier keypoints on the corners.
        If you find this sentence difficult to understand, you can go to check the demo.
        In the demo, a corresponding texel is obtained from 3 points. The 4th point is predicted from the 3 points.


    Args:
        img: Input RGB image
        proposal: Outputs from get_proposal(). Proposal is a list of inliers.
        texel_size: The patch size (U, V) of the patch transformed from the quadrilateral.
                    In this implementation, U is equal to V. (U = V = texel_size = 50.) The texel is a square.
    Returns:
        texels: A numpy ndarray of the shape (#texels, texel_size, texel_size, #channels).
    """
    # YOUR CODE HERE

    # END
    return texels


def score_proposal(texels, a_score_count_min=3):
    """
    Calcualte A-Score.

    Hints:
    (1) Each channel is normalized separately.
        The A-score for a RGB texel is the average of 3 A-scores of each channel.

    (2) You can return 1000 (in our example) to denote a invalid A-score.
        An invalid A-score is usually results from clusters with less than "a_score_count_min" texels.

    Args:
        texels: A numpy ndarray of the shape (#texels, window, window, #channels).
        a_score_count_min: Minimal number of texels we need to calculate the A-score.
    Returns:
        a_score: A-score calculated from the texels. If there are no sufficient texels, return 1000.
    """

    K, U, V, C = texels.shape

    # YOUR CODE HERE

    # END

    return a_score


# Part 3
# You are free to change the input argument of the functions in Part 3.
# GIVEN
def non_max_suppression(response, suppress_range, threshold=None):
    """
    Non-maximum Suppression for translation symmetry detection

    The general approach for non-maximum suppression is as follows:
        1. Perform thresholding on the input response map. Set the points whose values are less than the threshold as 0.
        2. Find the largest response value in the current response map
        3. Set all points in a certain range around this largest point to 0.
        4. Save the current largest point
        5. Repeat the step from 2 to 4 until all points are set as 0.
        6. Return the saved points are the local maximum.

    Args:
        response: numpy.ndarray, output from the normalized cross correlation
        suppress_range: a tuple of two ints (H_range, W_range). The points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    Returns:
        threshold: int, points with value less than the threshold are set to 0
    """
    H, W = response.shape[:2]
    H_range, W_range = suppress_range
    res = np.copy(response)

    if threshold is not None:
        res[res < threshold] = 0

    idx_max = res.reshape(-1).argmax()
    x, y = idx_max // W, idx_max % W
    point_set = set()
    while res[x, y] != 0:
        point_set.add((x, y))
        res[max(x - H_range, 0): min(x+H_range, H),
            max(y - W_range, 0): min(y+W_range, W)] = 0
        idx_max = res.reshape(-1).argmax()
        x, y = idx_max // W, idx_max % W
    for x, y in point_set:
        res[x, y] = response[x, y]
    return res


def template_match(img, proposal, threshold):
    """
    Perform template matching on the original input image.

    Hints:
    (1) You may find cv2.copyMakeBorder and cv2.matchTemplate helpful. The cv2.copyMakeBorder is used for padding.
        Alternatively, you can use your implementation in Lab 1 for template matching.

    (2) For non-maximum suppression, you can either use the one you implemented for lab 1 or the code given above.

    Returns:
        response: A sparse response map from non-maximum suppression.
    """
    # YOUR CODE HERE

    # END
    return response


def maxima2grid(img, proposal, response):
    """
    Estimate 4 lattice points from each local maxima.

    Hints:
    (1) We can transfer the 4 offsets between the center of the original template and 4 lattice unit points to new detected centers.

    Args:
        response: The response map from `template_match()`.

    Returns:
        points_grid: an numpy ndarray of shape (N, 2), where N is the number of grid points.

    """
    # YOUR CODE HERE

    # END

    return points_grid


def refine_grid(img, proposal, points_grid):
    """
    Refine the detected grid points.

    Args:
        points_grid: The output from the `maxima2grid()`.

    Returns:
        points: A numpy ndarray of shape (N, 2), where N is the number of refined grid points.
    """
    # YOUR CODE HERE

    # END

    return points


def grid2latticeunit(img, proposal, points):
    """
    Convert each lattice grid point into integer lattice grid.

    Hints:
    (1) Since it is difficult to know whether two points should be connected, one way is to map each point into an integer position.
        The integer position should maintain the spatial relationship of these points.
        For instance, if we have three points x1=(50, 50), x2=(70, 50) and x3=(70, 70), we can map them (4, 5), (5, 5) and (5, 6).
        As the distances between (4, 5) and (5, 5), (5, 5) and (5, 6) are both 1, we know that (x1, x2) and (x2, x3) form two edges.

    (2) You can use affine transformation to build the mapping above, but do not perform global affine transformation.

    (3) The mapping in the hints above are merely to know whether two points should be connected.
        If you have your own method for finding the relationship, feel free to implement your owns and ignore the hints above.


    Returns:
        edges: A list of edges in the lattice structure. Each edge is defined by two points. The point coordinate is in the image coordinate.
    """

    # YOUR CODE HERE

    # END

    return edges
