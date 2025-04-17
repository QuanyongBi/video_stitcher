import cv2
import numpy as np
import os

def extract_and_save_frames(video_path, output_dir = "extracted_frames", interval = 1):
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
            
            if (frame_idx + 1) % 10 == 0:
                print("Extracted and saved frame", frame_idx + 1, "from", video_path)
                
            frame_idx += 1
        
        count += 1
            
    cap.release()
    print("Extracted", len(frames), "frames from", video_path)
    return frames, saved_paths

extract_and_save_frames("video_data/video5/forest1.mp4")