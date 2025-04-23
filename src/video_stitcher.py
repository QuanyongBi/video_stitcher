import cv2
import numpy as np
from feature_matching import detect_and_match_features, find_transformation
from output_visualize import visualize_output

def stitch_images_linear(frames, debug_progress = False):
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
    
    # T_scale = np.array([
    #     [2, 0, 0],
    #     [0, 2, 0],
    #     [0, 0, 1]
    # ], dtype = float)
    # T_scale_inv = np.linalg.inv(T_scale)
    
    transforms = {mid_idx: T_ref}
    
    # Using middle out lol
    for i in range(mid_idx + 1, n):
        print(f"Progress: {((i - (mid_idx + 1)) / (len(frames) - 1) * 100.0):.2f}%")
        
        kp_prev, _, kp_cur, _, matches = detect_and_match_features(frames[i], frames[i-1], feature_num=10000)

        if matches is None or len(matches) < 5:
            print("Not enough matches between images", i - 1, "and", i)
            continue # we can also try ending but not sure yet
        
        H, _ = find_transformation(kp_prev, kp_cur, matches)
        
        if H is None:
            print("Could not find homography for image", i)
            continue
        
        # Chain the homography matrix with multiplication
        T_prev = transforms[i-1]
        # H_scaled = T_scale @ H @ T_scale_inv
        # T_cur = T_prev @ H
        T_cur = T_prev @ H
        transforms[i] = T_cur
        
        # Warping image
        img = frames[i]
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
             
        kp_next, _, kp_cur, _, matches = detect_and_match_features(frames[i], frames[i+1], feature_num=10000)

        if matches is None or len(matches) < 5:
            print("Not enough matches between images", i + 1, "and", i)
            break
        
        H, _ = find_transformation(kp_next, kp_cur, matches)
        
        if H is None:
            print("Could not find homography for image", i)
            break
        
        # Chain the homography matrix with multiplication
        T_next = transforms[i+1]
        # H_scaled = T_scale @ H @ T_scale_inv
        T_cur = T_next @ H
        # T_cur = T_next @ H_scaled
        transforms[i] = T_cur
        
        # Warping image
        img = frames[i]
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

def stitch_images_stack(frames):
    reference_panorama = frames[0].copy()
    
    for i in range(1, len(frames)):
        print(f"Processing frame {i}/{len(frames)-1} ({i/(len(frames)-1)*100:.2f}%)")
        cur_frame = frames[i]
        reference_panorama = stitch_two_frames(reference_panorama, cur_frame)
        # Feature extraction and matching
        
    return reference_panorama


def stitch_images_divide_conquer(frames):
    print("cur_recurrence level: ", len(frames))
    if(len(frames) == 0):
        return None
    if(len(frames) == 1):
        return frames[0]
    if(len(frames) == 2):
        return stitch_two_frames(frames[0], frames[1])
    
    mid = len(frames) // 2
    left_res = stitch_images_divide_conquer(frames[:mid])
    right_res = stitch_images_divide_conquer(frames[mid+1:])
        
    return stitch_two_frames(left_res, right_res)
    

def stitch_two_frames(reference_panorama, cur_frame):
    kp_cur, _, kp_prev, _, matches = detect_and_match_features(frame_cur=cur_frame, frame_prev=reference_panorama, feature_num=20000)
    # Find homography matrix
    H, _ = find_transformation(kp_cur, kp_prev, matches)
        
    # Create a new canvas based on the new image
    canvas_w, canvas_h, offset_x, offset_y = calculate_canvas_size(reference_panorama, cur_frame, H)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
    # Place the original panorama on the new canvas
    h_ref, w_ref = reference_panorama.shape[:2]
    canvas[offset_y:offset_y+h_ref, offset_x:offset_x+w_ref] = reference_panorama
        
    Translation = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)
    H_combined = Translation @ H
    
    warped_cur_frame = np.zeros_like(canvas)    
    warped_cur_frame = cv2.warpPerspective(
        cur_frame, 
        H_combined, 
        (canvas_w, canvas_h), 
        dst=warped_cur_frame,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )
        
    # Create a mask of the warped frame and combine the two frames
    warped_mask = (cv2.cvtColor(warped_cur_frame, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    canvas[warped_mask > 0] = warped_cur_frame[warped_mask > 0]
    reference_panorama = canvas
    return reference_panorama


def calculate_canvas_size(reference_panorama, cur_frame, H):
    h_ref, w_ref = reference_panorama.shape[:2]
    h_cur, w_cur = cur_frame.shape[:2]
    
    # Get corners of current frame
    corners = np.array([
        [0, 0, 1],          # top-left
        [w_cur, 0, 1],      # top-right
        [w_cur, h_cur, 1],  # bottom-right
        [0, h_cur, 1]       # bottom-left
    ], dtype=np.float32).T
    
    # Transform corners through homography
    transformed_corners = H @ corners
    # Normalize homogeneous coordinates
    transformed_corners /= transformed_corners[2]
    
    # Find min/max x and y coordinates - use floor/ceil to ensure coverage
    min_x = int(min(0, np.floor(transformed_corners[0].min())))
    min_y = int(min(0, np.floor(transformed_corners[1].min())))
    max_x = int(max(w_ref, np.ceil(transformed_corners[0].max())))
    max_y = int(max(h_ref, np.ceil(transformed_corners[1].max())))
    
    # Add padding
    padding = 0  # Add some buffer
    width = max_x - min_x + padding*2
    height = max_y - min_y + padding*2
    
    # Calculate offset for the reference panorama
    offset_x = abs(min_x) + padding if min_x < 0 else padding
    offset_y = abs(min_y) + padding if min_y < 0 else padding
    
    return width, height, offset_x, offset_y
