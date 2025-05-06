import cv2
import numpy as np
from feature_matching import detect_and_match_features, find_transformation
from video_stitcher import calculate_canvas_size
from output_visualize import visualize_output

def stitch_images_stack(frames, corrected_frames):
    ref_pano = frames[0]
    corrected_pano = corrected_frames[0]
    for i in range(1, len(frames)):
        print(f"Processing frame {i}/{len(frames)-1} ({i/(len(frames)-1)*100:.2f}%)")
        ref_pano, corrected_pano = stitch_two_frames(ref_pano, corrected_pano, frames[i], corrected_frames[i])
    return ref_pano, corrected_pano

def stitch_two_frames(ref_pano_highres, corrected_pano_highres, cur_frame_highres, corrected_frame_highres, feature_num=10000, scale_factor=1, feather_width=30):
    # Create low resolution versions for the color_corrected pano and frame
    h, w = corrected_pano_highres.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    corrected_pano_lowres = cv2.resize(corrected_pano_highres, (new_w, new_h))
    
    h, w = corrected_frame_highres.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    corrected_frame_lowres = cv2.resize(corrected_frame_highres, (new_w, new_h))
    
    # Get dimensions to calculate scaling factors
    h_ref_high, w_ref_high = corrected_pano_highres.shape[:2]
    h_ref_low, w_ref_low = corrected_pano_lowres.shape[:2]
    # Calculate scaling factors
    scale_factor_x = w_ref_high / w_ref_low
    scale_factor_y = h_ref_high / h_ref_low
    
    kp_cur, _, kp_prev, _, matches = detect_and_match_features(
        frame_cur=corrected_frame_lowres, 
        frame_prev=corrected_pano_lowres, 
        feature_num=feature_num
    )
    
    # Find homography matrix for low-res corrected images
    H_lowres, status = find_transformation(kp_cur, kp_prev, matches)
    if H_lowres is None:
        print("Could not find homography, returning original reference panorama")
        return ref_pano_highres, corrected_pano_highres
    
    # Create scaling matrices
    scale_matrix = np.array([
        [scale_factor_x, 0, 0],
        [0, scale_factor_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    scale_matrix_inv = np.array([
        [1/scale_factor_x, 0, 0],
        [0, 1/scale_factor_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Scale the homography matrix: H_highres = S * H_lowres * S^(-1)
    H_highres = scale_matrix @ H_lowres @ scale_matrix_inv
    
    # Create a new canvas based on the high-res images
    canvas_w, canvas_h, offset_x, offset_y = calculate_canvas_size(ref_pano_highres, cur_frame_highres, H_highres)
    canvas_ref = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_corrected = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    h_ref, w_ref = ref_pano_highres.shape[:2]
    canvas_ref[offset_y:offset_y+h_ref, offset_x:offset_x+w_ref] = ref_pano_highres
    canvas_corrected[offset_y:offset_y+h_ref, offset_x:offset_x+w_ref] = corrected_pano_highres
    
    # Create a mask for the reference panorama
    ref_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    ref_mask[offset_y:offset_y+h_ref, offset_x:offset_x+w_ref] = 255
    
    # Find contours of the reference panorama to identify true edges
    contours_ref, _ = cv2.findContours(ref_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_edge_mask = np.zeros_like(ref_mask)
    cv2.drawContours(ref_edge_mask, contours_ref, -1, 255, 1)  # Draw just the edge contour (1 pixel thick)
    
    Translation = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H_combined = Translation @ H_highres
    
    # Warp the high-res current frame
    warped_cur_frame = np.zeros_like(canvas_ref)
    warped_cur_frame = cv2.warpPerspective(
        cur_frame_highres, 
        H_combined, 
        (canvas_w, canvas_h), 
        dst=warped_cur_frame,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )
    
    warped_corrected_frame = np.zeros_like(canvas_corrected)
    warped_corrected_frame = cv2.warpPerspective(
        corrected_frame_highres, 
        H_combined, 
        (canvas_w, canvas_h), 
        dst=warped_corrected_frame,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )
    
    
    warped_mask = cv2.cvtColor(warped_corrected_frame, cv2.COLOR_BGR2GRAY) > 0
    canvas_corrected[warped_mask] = warped_corrected_frame[warped_mask]
    
    mask2 = cv2.threshold(cv2.cvtColor(warped_cur_frame, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_TOZERO)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, kernel)
    
    # warped_cur_frame[mask2==0] = 0

    # Apply the mask
    warped_cur_frame[mask2 > 0] = warped_cur_frame[mask2 > 0]
    canvas_ref[mask2> 0] = warped_cur_frame[mask2 > 0]
    
    return canvas_ref, canvas_corrected