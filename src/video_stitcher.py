import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def extract_and_save_frames(video_path, output_dir = "data/extracted_frames", interval = 1):
    # Create output directory if it doesn't exist
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok = True)
    
    frames = []
    saved_paths = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frames based on interval
        if count % interval == 0:  
            # Save the frame
            filename = "frame_" + str(frame_idx) + ".png"
            filepath = os.path.join(frames_dir, filename)
            cv2.imwrite(filepath, frame)
            
            # Store frame and path
            frames.append(frame)
            saved_paths.append(filepath)
                
            frame_idx += 1
        
        count += 1
            
    cap.release()
    print("Extracted", len(frames), "frames from", video_path)
    return frames, saved_paths

frames, paths = extract_and_save_frames("data/video_data/video5/forest1.mp4", "data/extracted_frames", 10)

def detect_and_match_features(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error loading images from {img1_path} or {img2_path}")
        return None
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=1000)
    
    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)
    
    print(f"Image 1: {len(kp1)} keypoints detected")
    print(f"Image 2: {len(kp2)} keypoints detected")
    
    # Currently using Brute Force Matching
    # To be Optimize
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches")
    
    # # Draw matches
    # match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # # Display results
    # plt.figure(figsize=(16, 8))
    # plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    # plt.title(f'{len(good_matches)} Matches Found')
    # plt.tight_layout()
    # plt.show()
    
    return kp1, desc1, kp2, desc2, good_matches

kp1, desc1, kp2, desc2, matches = detect_and_match_features(paths[1], paths[0])

def find_transformation(kp1, kp2, matches, method='homography'):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, status

def create_canvs(img1_path, img2_path, H):
    img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    canvas_h = h1 * 3
    canvas_w = w2 * 3
    
    # Define the origin (center of the canvas)
    origin_x = canvas_w // 2
    origin_y = canvas_h // 2
    
    # Calculate offset for img1 to place its center at origin
    img1_offset_x = origin_x - (w1 // 2)
    img1_offset_y = origin_y - (h1 // 2)
    
    # Create empty canvas
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    # Add red border to the first image
    img1_with_border = img1.copy()
    img1_with_border[0:3, :] = [0, 0, 255]  # Top border
    img1_with_border[:, 0:3] = [0, 0, 255]  # Left border
    img1_with_border[h1-3:h1, :] = [0, 0, 255]  # Bottom border
    img1_with_border[:, w1-3:w1] = [0, 0, 255]  # Right border
    # Place the first image on the canvas
    panorama[img1_offset_y:img1_offset_y+h1, img1_offset_x:img1_offset_x+w1] = img1_with_border
    
    T1 = np.array([
        [1, 0, img1_offset_x],
        [0, 1, img1_offset_y],
        [0, 0, 1]
    ])
    H_inv = np.linalg.inv(H)
    transform_canvas_to_img2 = np.dot(H_inv, np.linalg.inv(T1))
    img2_transformed = cv2.warpPerspective(img2, 
                                          np.linalg.inv(transform_canvas_to_img2), 
                                          (canvas_w, canvas_h))
    _, mask = cv2.threshold(cv2.cvtColor(img2_transformed, cv2.COLOR_BGR2GRAY), 
                            1, 255, cv2.THRESH_BINARY)
    # Replace the pixels in the panorama with the transformed img2
    # For simplicity, we'll just overwrite the pixels without blending
    panorama[mask > 0] = img2_transformed[mask > 0]
    
    non_black = np.where(panorama > 0)
    if len(non_black[0]) > 0:
        y_min, y_max = np.min(non_black[0]), np.max(non_black[0])
        x_min, x_max = np.min(non_black[1]), np.max(non_black[1])
        
        # Add some padding
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(canvas_h - 1, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(canvas_w - 1, x_max + padding)
        
        # Crop the image
        panorama = panorama[y_min:y_max+1, x_min:x_max+1]
    
    # Display results
    plt.figure(figsize=(20, 10))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title('Stitched Panorama')
    
    plt.tight_layout()
    plt.show()
    return result


H, status = find_transformation(kp1, kp2, matches)
print("H=", H)
result = create_canvs(paths[0], paths[1], H)