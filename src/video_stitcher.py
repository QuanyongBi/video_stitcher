import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def extract_and_save_frames(video_path, output_dir = "data/extracted_frames", interval = 1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_idx = 0
    
    if not cap.isOpened():
        print("Wrong directory or some weird stuff happened:", video_path)
    else:
        # Create output directory if it doesn't exist
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(output_dir, video_name)
        os.makedirs(frames_dir, exist_ok = True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frames based on interval
            if count % interval == 0:  
                # Save the frame
                # filename = "frame_" + str(frame_idx) + ".png"
                # filepath = os.path.join(frames_dir, filename)
                # cv2.imwrite(filepath, frame)
                
                # Store frame and path
                frames.append(frame)
                # saved_paths.append(filepath)
                    
                frame_idx += 1
            
            count += 1
                
        cap.release()
        print("Extracted", len(frames), "frames from", video_path)
    return frames

def detect_and_match_features(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=5000)
    
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

def find_transformation(kp1, kp2, matches, method='homography'):
    if len(matches) < 4:
        print("Not enough matches to find homography")
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, status

def stitch_images(frames, debug_progress = False):
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
        T_cur = T_prev @ H
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
        T_cur = T_next @ H
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
   
def visualize_output(frames, output):
    
    # Show original images - too much... not needed
    # n_images = len(image_paths)
    # for i, path in enumerate(image_paths):
    #     img = cv2.imread(path)
    #     ax = plt.subplot(2, n_images, i+1)
    #     ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     ax.set_title(f'Image {i}')
    #     ax.axis('off')
    
    # Show output
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax.set_title('Stitched Panorama')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def compress_video(input_path, output_path, scale = 0.5, codec = "mp4v"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")
    
    # Setting properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = int(src_width  * scale)
    out_h = int(src_height * scale)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (out_w, out_h))
    
    print(f"Compressing {input_path} to {output_path}")
    print(f"{src_width}×{src_height}@{src_fps:.2f}fps to {out_w}×{out_h}@{src_fps:.2f}fps")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame
        small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(small)
    
    cap.release()
    writer.release()
    print("Done Compressing!")
    
    
def main():
    # Extract frames from video
    video_path = "data/video_data/video2/real_002.mp4"
    compress_video(video_path,
               "data/compressed/hd_halfres.mp4",
               scale=0.5,
               codec='mp4v')
    frames = extract_and_save_frames("data/compressed/hd_halfres.mp4", "data/extracted_frames", 1)
    
    # Visualizing the output
    output = stitch_images(frames, False)
    if output is not None:
        visualize_output(frames, output)
        
if __name__ == "__main__":
    main()

# def create_canvs(img1_path, img2_path, H):
#     img1, img2 = cv2.imread(img1_path), cv2.imread(img2_path)
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
    
#     canvas_h = h1 * 3
#     canvas_w = w2 * 3
    
#     # Define the origin (center of the canvas)
#     origin_x = canvas_w // 2
#     origin_y = canvas_h // 2
    
#     # Calculate offset for img1 to place its center at origin
#     img1_offset_x = origin_x - (w1 // 2)
#     img1_offset_y = origin_y - (h1 // 2)
    
#     # Create empty canvas
#     panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
#     # Add red border to the first image
#     img1_with_border = img1.copy()
#     img1_with_border[0:3, :] = [0, 0, 255]  # Top border
#     img1_with_border[:, 0:3] = [0, 0, 255]  # Left border
#     img1_with_border[h1-3:h1, :] = [0, 0, 255]  # Bottom border
#     img1_with_border[:, w1-3:w1] = [0, 0, 255]  # Right border
#     # Place the first image on the canvas
#     panorama[img1_offset_y:img1_offset_y+h1, img1_offset_x:img1_offset_x+w1] = img1_with_border
    
#     T1 = np.array([
#         [1, 0, img1_offset_x],
#         [0, 1, img1_offset_y],
#         [0, 0, 1]
#     ])
#     H_inv = np.linalg.inv(H)
#     transform_canvas_to_img2 = np.dot(H_inv, np.linalg.inv(T1))
#     img2_transformed = cv2.warpPerspective(img2, 
#                                           np.linalg.inv(transform_canvas_to_img2), 
#                                           (canvas_w, canvas_h))
#     _, mask = cv2.threshold(cv2.cvtColor(img2_transformed, cv2.COLOR_BGR2GRAY), 
#                             1, 255, cv2.THRESH_BINARY)
#     # Replace the pixels in the panorama with the transformed img2
#     # For simplicity, we'll just overwrite the pixels without blending
#     panorama[mask > 0] = img2_transformed[mask > 0]
    
#     non_black = np.where(panorama > 0)
#     if len(non_black[0]) > 0:
#         y_min, y_max = np.min(non_black[0]), np.max(non_black[0])
#         x_min, x_max = np.min(non_black[1]), np.max(non_black[1])
        
#         # Add some padding
#         padding = 10
#         y_min = max(0, y_min - padding)
#         y_max = min(canvas_h - 1, y_max + padding)
#         x_min = max(0, x_min - padding)
#         x_max = min(canvas_w - 1, x_max + padding)
        
#         # Crop the image
#         panorama = panorama[y_min:y_max+1, x_min:x_max+1]
    
#     # Display results
#     plt.figure(figsize=(20, 10))
    
#     plt.subplot(131)
#     plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
#     plt.title('Image 1')
    
#     plt.subplot(132)
#     plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
#     plt.title('Image 2')
    
#     plt.subplot(133)
#     plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
#     plt.title('Stitched Panorama')
    
#     plt.tight_layout()
#     plt.show()
#     return result


# H, status = find_transformation(kp1, kp2, matches)
# print("H=", H)
# result = create_canvs(paths[0], paths[1], H)
