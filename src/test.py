from video_utils import extract_and_save_frames
from output_visualize import visualize_output
from video_stitcher import stitch_images_linear, stitch_images_stack, stitch_images_divide_conquer
from color_alignment import correct_all_frames
import cv2

def main():
    # Extract frames from video
    video_paths = ["data/video_data/video1/real_001.mp4"]
    frames = extract_and_save_frames(video_paths, "data/extracted_frames", 5)
    # corrected = correct_all_frames(frames, 0)
    # corrected_low_res = correct_all_frames(low_res_frames, 0)
    
    # Visualizing the output
    output = stitch_images_stack(frames)
    if output is not None:
        visualize_output(output)
    
    # output = stitch_images_divide_conquer(frames)
    # # output = stitch_images_divide_conquer(frames)
    # if output is not None:
    #     visualize_output(output)
    
    # image_stitcher = cv2.Stitcher_create()
    # err, output = image_stitcher.stitch(frames)
    # if not err:
    #     visualize_output(output)
    
    return 
        
if __name__ == "__main__":
    main()