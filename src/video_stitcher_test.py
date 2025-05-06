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

def stitch_two_frames(ref_pano_highres, corrected_pano_highres, cur_frame_highres, corrected_frame_highres, feature_num=10000, scale_factor=0.75, feather_width=30):
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
    
    # Create warped mask
    initial_mask = cv2.cvtColor(warped_cur_frame, cv2.COLOR_BGR2GRAY) > 0
    warped_mask_uint8 = initial_mask.astype(np.uint8) * 255
    
    # Find the external contour of the warped frame
    contours, _ = cv2.findContours(warped_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a clean mask containing only the external contours
    outer_mask = np.zeros_like(warped_mask_uint8)
    cv2.drawContours(outer_mask, contours, -1, 255, -1)  # Fill all external contours
    visualize_output(outer_mask)
    
    # Create a mask showing just the edge of the warped frame
    warped_edge_mask = np.zeros_like(warped_mask_uint8)
    cv2.drawContours(warped_edge_mask, contours, -1, 255, 1)  # Draw just the edge contour (1 pixel thick)
    
    # For the corrected canvas, use the original hard blending method
    # But use the outer mask (which ignores internal black regions)
    canvas_corrected[outer_mask > 0] = warped_corrected_frame[outer_mask > 0]
    warped_mask = outer_mask > 0
    
    # Find overlap region using the contour-based masks
    overlap_mask = (ref_mask > 0) & warped_mask
    
    if np.any(overlap_mask):
        # We'll use a narrower feathering width for gentler transitions
        narrow_feather_width = min(50, feather_width)  # Reduce feathering width
        
        # Create clean distance maps using the contour-based outer mask
        dist_from_warped_edge = cv2.distanceTransform(outer_mask, cv2.DIST_L2, 3)
        dist_from_ref_edge = cv2.distanceTransform(ref_mask, cv2.DIST_L2, 3)
        
        # Create binary masks for the thin boundary regions
        warped_edge_region = (dist_from_warped_edge <= narrow_feather_width) & warped_mask
        ref_edge_region = (dist_from_ref_edge <= narrow_feather_width) & (ref_mask > 0)
        
        # Identify the edge overlap region - where both edge regions intersect
        edge_overlap = warped_edge_region & ref_edge_region & overlap_mask
        
        # Create weight maps only for the edge overlap regions
        # For warped frame edge
        warped_weight = dist_from_warped_edge.copy()
        warped_weight[warped_weight > narrow_feather_width] = narrow_feather_width
        warped_weight = warped_weight / narrow_feather_width  # Normalize to [0,1]
        
        # For reference frame edge
        ref_weight = dist_from_ref_edge.copy()
        ref_weight[ref_weight > narrow_feather_width] = narrow_feather_width
        ref_weight = ref_weight / narrow_feather_width  # Normalize to [0,1]
        
        # Calculate the final weight map for the edge overlap
        # Higher values favor the warped frame (new frame)
        edge_weight_map = np.ones_like(warped_weight)  # Default to 1 (prefer new frame)
        
        # In the edge overlap region, adjust weights based on distance from edges
        # This will create a smoother transition
        edge_weight_map[edge_overlap] = (warped_weight[edge_overlap] + 0.5) / (warped_weight[edge_overlap] + ref_weight[edge_overlap] + 0.5)
        
        # Apply the weight map to all regions:
        # 1. In central overlap: use warped frame (new frame) exclusively
        # 2. In edge overlap: use the calculated weight map for smooth transition
        central_overlap = overlap_mask & ~edge_overlap
        
        for c in range(3):  # For each color channel
            # In the edge overlap region, blend based on the weight map
            canvas_ref[:,:,c][edge_overlap] = (
                (1 - edge_weight_map[edge_overlap]) * canvas_ref[:,:,c][edge_overlap] + 
                edge_weight_map[edge_overlap] * warped_cur_frame[:,:,c][edge_overlap]
            ).astype(np.uint8)
            
            # In the central overlap region, use the warped frame (new frame)
            canvas_ref[:,:,c][central_overlap] = warped_cur_frame[:,:,c][central_overlap]
    
        # For non-overlapping regions of the warped frame, just copy directly
        non_overlap_warped = warped_mask & ~overlap_mask
        canvas_ref[non_overlap_warped] = warped_cur_frame[non_overlap_warped]
    else:
        # If no overlap, simply copy the warped frame using the outer mask
        canvas_ref[warped_mask] = warped_cur_frame[warped_mask]
    
    return canvas_ref, canvas_corrected