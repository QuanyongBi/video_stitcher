from video_utils import compress_video, extract_and_save_frames
from output_visualize import visualize_output
from video_stitcher import stitch_images

def main():
    # Extract frames from video
    video_path = "data/video_data/video1/real_001.mp4"
    compress_video(video_path,
               "data/compressed/hd_halfres.mp4",
               scale=0.5,
               codec='mp4v')
    high_res_frames = extract_and_save_frames("data/video_data/video1/real_001.mp4", "data/extracted_frames", 16)
    frames = extract_and_save_frames("data/compressed/hd_halfres.mp4", "data/extracted_frames", 16)
    
    # Visualizing the output
    output = stitch_images(frames, high_res_frames, False)
    if output is not None:
        visualize_output(frames, output)
        
if __name__ == "__main__":
    main()