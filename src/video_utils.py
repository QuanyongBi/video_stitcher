import cv2
import os
import numpy as np

def compress_video(input_path, output_path, scale=0.5, codec="mp4v"):
    """
    Compress a video by resizing its frames.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path to save the compressed video
        scale (float): Scale factor for resizing (default: 0.5)
        codec (str): Four character code for video codec (default: 'mp4v')
    
    Returns:
        bool: True if compression was successful, False otherwise
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")
    
    # Setting properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = int(src_width * scale)
    out_h = int(src_height * scale)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (out_w, out_h))
    
    print(f"Compressing {input_path} to {output_path}")
    print(f"{src_width}×{src_height}@{src_fps:.2f}fps to {out_w}×{out_h}@{src_fps:.2f}fps")
    
    success = True
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
    return success


def extract_and_save_frames(video_paths, output_dir = "data/extracted_frames", interval = 1):
    frames = []
    
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
    print("Extracted", len(frames), "frames from", video_paths)
    return frames
