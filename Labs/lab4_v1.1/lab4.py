import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift  # added
from scipy.spatial.distance import cdist  # added
from heapq import heappop, heappush, heapify  # added

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
    pts = []
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
                for patch_p in patch_pts:
                    # goodFeaturestoTrack returns (x, y) coordinates
                    # need to swap the order to return (y, x) coordinates
                    x, y = patch_p.ravel()
                    pts.append(np.array([y, x]).astype('float64'))
    pts = np.array(pts)
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
    features = []
    for p in pts:
        y, x = p.astype(int)
        patch = img_gray[y - window_patch: y + window_patch + 1,
                         x - window_patch: x + window_patch + 1]
        # normalize the patch
        normalized_patch = (patch - np.mean(patch)) / np.std(patch)
        features.append(normalized_patch.flatten())
    features = np.array(features).astype('float64')
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
    # centroids = {}
    # for i in range(no_of_features):
    #     centroids[i] = features[i]
    # while True:
    #     converged = True
    #     for f_idx, centroid in centroids.items():
    #         within_bandwidth = []
    #         for feature in features:
    #             if np.linalg.norm(feature - centroid) < bandwidth:
    #                 within_bandwidth.append(feature)
    #         new_centroid = np.mean(within_bandwidth, axis=0)
    #         # if np.linalg.norm(centroid - new_centroid) > 0:
    #         if not np.array_equal(centroid, new_centroid):
    #             converged = False
    #             centroids.update({f_idx: new_centroid})
    #     if converged:
    #         break
    # unique_centroids = np.unique(np.array(list(centroids.values())), axis=0)
    # labels = []
    # for centroid in centroids.values():
    #     idx = np.where(unique_centroids == centroid)[0][0]
    #     labels.append(idx)
    # labels = np.array(labels)
    # ===============================================================
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(features)
    cluster_centers = ms.cluster_centers_
    labels = ms.labels_
    clustering = {'cluster_centers_': cluster_centers,
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
        candidate_clusters[labels[i]].append(i)
    for cluster in candidate_clusters:
        cluster_size = len(cluster)
        if cluster_size < tau1:  # discard small clusters
            print('discarding small cluster...')
            continue
        elif cluster_size > tau2:  # partition large clusters
            print('partitioning big cluster...')
            kmeans = KMeans(n_clusters=(cluster_size // tau2)
                            ).fit(features[cluster])
            kmeans_cluster_centers = kmeans.cluster_centers_
            kmeans_labels = kmeans.labels_
            no_of_kmeans_clusters = kmeans_cluster_centers.shape[0]
            kmeans_clusters = [[] for _ in range(no_of_kmeans_clusters)]
            for km_idx, f_idx in enumerate(cluster):
                kmeans_clusters[kmeans_labels[km_idx]].append(f_idx)
            for kmeans_cluster in kmeans_clusters:
                clusters.append(np.array(kmeans_cluster))
            continue
        else:
            print('adding new cluster...')
            clusters.append(np.array(cluster))
    clusters = [pts[cluster] for cluster in clusters]
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
    def top_k_nearest(pts, pt_0, k, pt_b=None, pt_c=None):
        heap = []
        heapify(heap)

        for i in range(len(pts)):
            pt_1 = pts[i]
            if pt_0[0] == pt_1[0] and pt_0[1] == pt_1[1]:
                continue
            if type(pt_b) is np.ndarray and pt_b[0] == pt_1[0] and pt_b[1] == pt_1[1]:
                continue
            if type(pt_c) is np.ndarray and pt_c[0] == pt_1[0] and pt_c[1] == pt_1[1]:
                continue

            dist = euclidean_distance(pt_0, pt_1)

            heappush(heap, (-1 * dist, i))

            if i > k:
                heappop(heap)

        result = []

        for i in range(len(heap)):
            idx = heap[i][1]
            result.append(pts[idx])

        return result

    def euclidean_distance(pt_0, pt_1):
        return math.sqrt((pt_0[0] - pt_1[0]) ** 2 + (pt_0[1] - pt_1[1]) ** 2)

    def transform_pts(pts, M):
        '''
        Performs the perspective transformation of coordinates

        Args:
            src (np.ndarray): Coordinates of points to transform (N,2)
            h_matrix (np.ndarray): Homography matrix (3,3)

        Returns:
            transformed (np.ndarray): Transformed coordinates (N,2)

        '''
        transformed = None

        input_pts = np.insert(pts, 2, values=1, axis=1)
        N = np.array([[0, 0, 1]])
        M = np.concatenate((M, N))

        transformed = np.zeros_like(input_pts)
        transformed = M.dot(input_pts.transpose())

        transformed = transformed[:-1]/transformed[-1]
        transformed = transformed.transpose().astype(np.float32)

        return transformed

    proposal = []

    dst_tri = np.array([[0, 0], [1, 0], [0, 1]]).astype(np.float32)

    max_count = 0

    for pt_0 in pts_cluster:
        # For each point, you choose 2 of the 10 nearest points
        nearest_10_pts = top_k_nearest(pts_cluster, pt_0, 10)

        for i in range(len(nearest_10_pts) - 1):
            for j in range(i + 1, len(nearest_10_pts)):
                pt_1 = nearest_10_pts[i]
                pt_2 = nearest_10_pts[j]

                # Define point a as the point to be the vertex opposite the longest side of the triangle formed by these three points
                dist_0_1 = euclidean_distance(pt_0, pt_1)
                dist_0_2 = euclidean_distance(pt_0, pt_2)
                dist_1_2 = euclidean_distance(pt_1, pt_2)

                a = [0, 0]
                b = [0, 0]
                c = [0, 0]
                if dist_0_1 < dist_1_2 and dist_0_2 < dist_1_2:
                    a = pt_0
                    b = pt_1
                    c = pt_2

                elif dist_0_1 < dist_0_2 and dist_1_2 < dist_0_2:
                    a = pt_1
                    b = pt_2
                    c = pt_0

                elif dist_0_2 < dist_0_1 and dist_1_2 < dist_0_1:
                    a = pt_2
                    b = pt_0
                    c = pt_1

                inliner_a = {
                    'pt_int': np.array([0, 0]).astype(int),
                    'pt': np.array(a).astype(int)
                }
                inliner_b = {
                    'pt_int': np.array([1, 0]).astype(int),
                    'pt': np.array(b).astype(int)
                }
                inliner_c = {
                    'pt_int': np.array([0, 1]).astype(int),
                    'pt': np.array(c).astype(int)
                }

                src_tri = np.array([a, b, c]).astype(np.float32)
                M = cv2.getAffineTransform(src_tri, dst_tri)

                nearest_X_pts = top_k_nearest(
                    pts_cluster, a, X, pt_b=b, pt_c=c)
                transformed_pts = transform_pts(nearest_X_pts, M)

                count = 0
                inliers = [inliner_a, inliner_b, inliner_c]

                for k in range(len(transformed_pts)):
                    original_pt = nearest_X_pts[k]
                    transformed_pt = transformed_pts[k]

                    nearest_grid = [
                        np.rint(transformed_pt[0]), np.rint(transformed_pt[1])]
                    dist = euclidean_distance(transformed_pt, nearest_grid)

                    if dist < tau_a:
                        pt_int = np.array(nearest_grid).astype(int)
                        pts_int = [inl['pt_int'] for inl in inliers]
                        pt = np.array(original_pt).astype(int)
                        # # do not add if the point projects to duplicate integer coordinates
                        # if inliers:
                        #     if any((pt_int == x).all() for x in pts_int):
                        #         continue
                        count += 1
                        inlier = {
                            'pt_int': pt_int,
                            'pt': pt
                        }
                        inliers.append(inlier)

                if count > max_count:
                    proposal = inliers
    # def transform_homography(src, w_matrix, getNormalized=True):
    #     '''
    #     Performs the perspective transformation of coordinates

    #     Args:
    #         src (np.ndarray): Coordinates of points to transform (N,2)
    #         h_matrix (np.ndarray): Homography matrix (3,3)

    #     Returns:
    #         transformed (np.ndarray): Transformed coordinates (N,2)

    #     '''
    #     transformed = None
    #     input_pts = np.insert(src, 2, values=1, axis=1)
    #     w_matrix = np.concatenate((w_matrix, [[0, 0, 1]]), axis=0)
    #     transformed = np.zeros_like(input_pts)
    #     transformed = w_matrix.dot(input_pts.transpose())
    #     if getNormalized:
    #         transformed = transformed[:-1]/transformed[-1]
    #     transformed = transformed.transpose().astype(np.float32)

    #     return transformed
    # proposal = []
    # distances = cdist(pts_cluster, pts_cluster, 'euclidean')
    # Y = 10
    # # iterate over all points in the given cluster
    # assert len(
    #     pts_cluster) >= Y, 'There are at least {} points in each cluster.'.format(Y)
    # max_n_inliers = -1
    # max_triplets = None
    # triplets_dst = np.array([[0, 0], [1, 0], [0, 1]]
    #                         ).astype(np.float32)
    # for p_idx, point in enumerate(pts_cluster):
    #     distance = distances[p_idx]
    #     Y_nearest_indices = (distance).argsort()[1: Y + 1 + 1]
    #     assert p_idx not in Y_nearest_indices, 'The {} nearest neighbors for a point should not contain itself.'.format(
    #         Y)
    #     # iterate over all possible YC2 combinations of triplets
    #     for i in range(Y):
    #         for j in range(Y):
    #             if i >= j:
    #                 continue
    #             # reorder triplets (maintain 'a' as the the vertex opposite to the longest side of the traingle formed by triplets)
    #             triplets = [
    #                 point,
    #                 pts_cluster[Y_nearest_indices[i]],
    #                 pts_cluster[Y_nearest_indices[j]]
    #             ]
    #             triplet_distances = cdist(triplets, triplets, 'euclidean')
    #             a_idx = np.argmin([np.max(triplet_distance)
    #                                for triplet_distance in triplet_distances])
    #             a = triplets.pop(a_idx)
    #             b = triplets.pop(0)
    #             c = triplets.pop(0)
    #             triplets = np.array([a, b, c]).astype(np.float32)
    #             warp_mat = cv2.getAffineTransform(triplets, triplets_dst)
    #             # count inliers (consider only the X nearest points)
    #             X_nearest_indices = (distance).argsort()[:X + 1]
    #             X_nearest_points = pts_cluster[X_nearest_indices]
    #             X_nearest_predicted = transform_homography(
    #                 X_nearest_points, warp_mat)
    #             inliers = [
    #                 {
    #                     'pt_int': [0, 0],
    #                     'pt': max_triplets[0]
    #                 },
    #                 {
    #                     'pt_int': [1, 0],
    #                     'pt': max_triplets[1]
    #                 },
    #                 {
    #                     'pt_int': [0, 1],
    #                     'pt': max_triplets[2]
    #                 }
    #             ]
    #             for X_idx, X_predicted in enumerate(X_nearest_predicted):
    #                 nearest_integer_position = np.rint(X_predicted)
    #                 X_distance = np.linalg.norm(
    #                     X_predicted - nearest_integer_position)
    #                 if X_distance < tau_a:
    #                     curr_n_inliers += 1
    #                     print(X_nearest_points[X_idx])
    #                     print(nearest_integer_position)
    #             blocker
    #             if curr_n_inliers > max_n_inliers:
    #                 # update the maximum number of inliers and the corresponding triplets
    #                 max_n_inliers = curr_n_inliers
    #                 max_triplets = triplets
    # proposal = None

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
    pts_int = np.array([p['pt_int'] for p in proposal])
    pts = np.array([p['pt'] for p in proposal])
    distances = cdist(pts_int, pts_int, 'euclidean')
    corners_indices = []
    for p_idx, pt_int in enumerate(pts_int):
        corners = []
        distance = distances[p_idx]
        candidates = np.where(distance == 1)[0]
        no_of_candidates = len(candidates)
        if no_of_candidates >= 2:
            for i in range(no_of_candidates - 1):
                for j in range(i + 1, no_of_candidates):
                    a = pt_int
                    b = pts_int[candidates[i]]
                    c = pts_int[candidates[j]]
                    if np.linalg.norm(b - c) == math.sqrt(2):
                        corners_indices.append(
                            [p_idx, candidates[i], candidates[j]])
    corners = np.array([pts[corner_indices] for corner_indices in corners_indices]
                       ).astype(np.float32)
    texels = []
    for corner in corners:
        point_fourth = corner[0] + \
            (corner[1] - corner[0]) + (corner[2] - corner[0])
        corner = np.concatenate((corner, [point_fourth]), axis=0)
        corner_dst = np.float32([[0,  0],
                                 [texel_size,  0],
                                 [0, texel_size],
                                 [texel_size, texel_size]])
        # print(corner)
        # print('\n')
        # transpose (h, w), as the input argument of cv2.getPerspectiveTransform is (w, h) ordering
        matrix_projective = cv2.getPerspectiveTransform(
            corner[:, [1, 0]], corner_dst)
        img_warped = cv2.warpPerspective(
            img, matrix_projective, (texel_size, texel_size))
        texels.append(img_warped)
    texels = np.array(texels)
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
    if K < a_score_count_min:  # check if at least 'a_score_count_min' number of texels
        return 1000.0
    # extract each channel from the source image
    channels_a_score = []
    for c_idx in range(3):
        # normalize the corresponding channel
        c_texels = [texel[:, :, c_idx] for texel in texels]
        c_texels_mean = np.mean(c_texels)
        c_texels_std = np.std(c_texels)
        for i, c_texel in enumerate(c_texels):
            c_texels[i] = (c_texel - c_texels_mean) / c_texels_std
        numerator = 0
        for u in range(U):
            for v in range(V):
                numerator += np.std([c_texel[u, v] for c_texel in c_texels])
        channel_a_score = numerator / (U * V * math.sqrt(K))
        channels_a_score.append(channel_a_score)
    a_score = np.mean(channels_a_score)
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
    # use the first three points to create the template
    pts = np.array([p['pt'] for p in proposal])
    a, b, c = pts[:3]
    d = a + (b-a) + (c-a)
    corner = np.array([a, b, c, d])
    max_y = np.max(corner[:, 0])
    min_y = np.min(corner[:, 0])
    max_x = np.max(corner[:, 1])
    min_x = np.min(corner[:, 1])
    template = img.copy()[min_y:max_y + 1, min_x:max_x + 1]
    # apply template matching
    result = cv2.matchTemplate(img.copy(), template, cv2.TM_CCOEFF_NORMED)
    response = non_max_suppression(result,
                                   (int(template.shape[0] * 0.8),
                                    int(template.shape[1] * 0.8)),
                                   threshold=threshold)
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
