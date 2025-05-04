from video_utils import compress_video, extract_and_save_frames
from output_visualize import visualize_output
from video_stitcher import stitch_images_linear, stitch_images_stack, stitch_images_divide_conquer
from color_alignment import correct_all_frames
import cv2

def main():
    # Extract frames from video
    video_paths = ["data/video_data/video5/forest1.mp4", "data/video_data/video5/forest2.mp4"]
    # video_path = "data/compressed/hd_halfres.mp4"
    # compress_video(video_path,
    #            output_path="data/compressed/hd_halfres.mp4",
    #            scale=0.5,
    #            codec='mp4v')
    # high_res_frames = extract_and_save_frames("data/video_data/custom/bedroom.MOV", "data/extracted_frames", 16)
    frames = extract_and_save_frames(video_paths, "data/extracted_frames", 5)
    # corrected = correct_all_frames(frames)
    
    # Visualizing the output
    output = stitch_images_stack(frames)
    if output is not None:
        visualize_output(output)
    
    # output = stitch_images_divide_conquer(corrected)
    # # output = stitch_images_divide_conquer(frames)
    # if output is not None:
    #     visualize_output(output)
    
    
    # image_stitcher = cv2.Stitcher_create()
    # err, output = image_stitcher.stitch(corrected)
    # if not err:
    #     visualize_output(output)
    
    return 
        
if __name__ == "__main__":
    main()