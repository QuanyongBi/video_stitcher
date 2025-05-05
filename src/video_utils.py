import cv2
import os
import numpy as np

def extract_and_save_frames(video_paths, output_dir = "data/extracted_frames", interval = 1, scale = 0.5):
    frames = []
    
    cap = cv2.VideoCapture(video_paths[0])
    if not cap.isOpened():
        raise IOError(f"Cannot open video files")
    
    for video_path in video_paths:
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
                    frames.append(frame)
                    frame_idx += 1
                
                count += 1
                    
            cap.release()
    print("Extracted", len(frames), "frames from", video_paths)
    return frames
