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
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display results
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'{len(good_matches)} Matches Found')
    plt.tight_layout()
    plt.show()
    
    return kp1, desc1, kp2, desc2, good_matches

if len(paths) >= 2:
    kp1, desc1, kp2, desc2, matches = detect_and_match_features(paths[0], paths[1])
else:
    print("Not enough frames were extracted. Make sure your video file exists and contains frames.")