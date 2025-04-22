import cv2
import numpy as np
from feature_matching import detect_and_match_features, find_transformation

def stitch_images(frames, high_res_frames, debug_progress = False):
    if len(frames) < 2:
        print("Not enough images to stitch")
        return None
    
    # We start with first image
    n = len(frames)
    mid_idx = n // 2
    ref_img = frames[mid_idx]
    h, w = ref_img.shape[:2]
    
    # Static canvas size, needs to be dynamic tho...
    canvas_h = h * 10
    canvas_w = w * 10
    
    output = np.zeros((canvas_h, canvas_w, 3), dtype = np.uint8)
    
    origin_x = canvas_w // 2 - w // 2
    origin_y = canvas_h // 2 - h // 2
    
    output[origin_y : origin_y + h, origin_x : origin_x + w] = ref_img
    
    # Transform reference matrix
    # No scaling needed!
    T_ref = np.array([
        [1, 0, origin_x],
        [0, 1, origin_y],
        [0, 0, 1]
    ], dtype = float)
    
    T_scale = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ], dtype = float)
    T_scale_inv = np.linalg.inv(T_scale)
    
    transforms = {mid_idx: T_ref}
    
    # Using middle out lol
    for i in range(mid_idx + 1, n):
        print(f"Progress: {((i - (mid_idx + 1)) / (len(frames) - 1) * 100.0):.2f}%")
        
        kp_prev, _, kp_cur, _, matches = detect_and_match_features(frames[i], frames[i - 1])

        if matches is None or len(matches) < 5:
            print("Not enough matches between images", i - 1, "and", i)
            continue # we can also try ending but not sure yet
        
        H, _ = find_transformation(kp_prev, kp_cur, matches)
        
        if H is None:
            print("Could not find homography for image", i)
            continue
        
        # Chain the homography matrix with multiplication
        T_prev = transforms[i-1]
        H_scaled = T_scale @ H @ T_scale_inv
        # T_cur = T_prev @ H
        T_cur = T_prev @ H_scaled
        transforms[i] = T_cur
        
        # Warping image
        img = high_res_frames[i]
        warped = np.zeros_like(output)
        warped = cv2.warpPerspective(img, T_cur, 
                                     (canvas_w, canvas_h), 
                                     dst=warped, 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_TRANSPARENT)
        
        mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        output[mask > 0] = warped[mask > 0]
        
        # if debug_progress:
        #     plt.figure(figsize = (15, 10))
        #     plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        #     plt.title(f'Output after adding image {i}')
        #     plt.show()
        
    for i in range(mid_idx - 1, -1, -1):
        print(f"Progress: {((((mid_idx) - i) / (len(frames) - 1) + 0.5) * 100.0):.2f}%")
        
        kp_next, _, kp_cur, _, matches = detect_and_match_features(frames[i], frames[i + 1])

        if matches is None or len(matches) < 5:
            print("Not enough matches between images", i + 1, "and", i)
            break
        
        H, _ = find_transformation(kp_next, kp_cur, matches)
        
        if H is None:
            print("Could not find homography for image", i)
            break
        
        # Chain the homography matrix with multiplication
        T_next = transforms[i+1]
        H_scaled = T_scale @ H @ T_scale_inv
        # T_cur = T_next @ H
        T_cur = T_next @ H_scaled
        transforms[i] = T_cur
        
        # Warping image
        img = high_res_frames[i]
        warped = np.zeros_like(output)
        warped = cv2.warpPerspective(img, T_cur, 
                                     (canvas_w, canvas_h), 
                                     dst=warped, 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_TRANSPARENT)
        
        mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        output[mask > 0] = warped[mask > 0]
        
        # if debug_progress:
        #     plt.figure(figsize = (15, 10))
        #     plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        #     plt.title(f'Output after adding image {i}')
        #     plt.show()
        
    return output
