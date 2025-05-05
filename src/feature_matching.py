import cv2
import numpy as np

def detect_and_match_features(frame_cur, frame_prev, feature_num):
    img1 = frame_cur
    img2 = frame_prev
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # orb = cv2.ORB_create(nfeatures=feature_num)
    sift = cv2.SIFT_create()
    
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    print(f"Image 1: {len(kp1)} keypoints detected")
    print(f"Image 2: {len(kp2)} keypoints detected")
    
    # Currently using Brute Force Matching
    # To be Optimize
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.match(desc1, desc2)
    # matches = bf.knnMatch(desc1, desc2, k=2)
    
    # FLANN parameters
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,       # Number of hash tables
    #                     key_size=12,            # Length of the key in bits
    #                     multi_probe_level=1)    # Number of levels to search in the hash tables
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # Number of times the trees in the index should be recursively traversed
    
    # Initiate the FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} good matches")
    return kp1, desc1, kp2, desc2, good_matches


def find_transformation(kp_src, kp_dest, matches, method='homography'):
    kp1 = kp_src
    kp2 = kp_dest
    
    if len(matches) < 4:
        print("Not enough matches to find homography")
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
    
    return H, status