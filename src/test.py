from video_utils import extract_and_save_frames
from output_visualize import visualize_output
from video_stitcher import stitch_images_stack, stitch_images_divide_conquer, stitch_two_frames
from color_alignment import correct_all_frames
import cv2

def main():
    # Extract frames from video
    video_paths = ["data/video_data/video8/clip1.mp4", "data/video_data/video8/clip2.mp4", "data/video_data/video8/clip3.mp4"]
    # frames = extract_and_save_frames(video_paths, 10)
    # corrected = correct_all_frames(frames, 0)
    # corrected_low_res = correct_all_frames(low_res_frames, 0)
    
    # Visualizing the output
    # output = stitch_images_stack(frames[0], frames)
    # if output is not None:
    #     visualize_output(output)
    
    # output = stitch_images_divide_conquer(frames)
    # # output = stitch_images_divide_conquer(frames)
    # if output is not None:
    #     visualize_output(output)
    
    # image_stitcher = cv2.Stitcher_create()
    # err, output = image_stitcher.stitch(frames)
    # if not err:
    #     visualize_output(output)
    
    # New stuff here...
    frames = extract_and_save_frames(video_paths[0], 5)
    output_pano_l = stitch_images_stack(frames[0:len(frames) // 2])
    output_pano_r = stitch_images_stack(frames[(len(frames) // 2 - 1):len(frames)])
    output_pano = stitch_two_frames(output_pano_r, output_pano_l, feature_num=75000)
    
    if output_pano is not None:
        for i in range(1, len(video_paths)):
            cur_vid = video_paths[i]
            frames = extract_and_save_frames(cur_vid, 5)
            output_pano_l = stitch_images_stack(frames[0:len(frames) // 2])
            output_pano_r = stitch_images_stack(frames[(len(frames) // 2 - 1):len(frames)])
            cur_pano = stitch_two_frames(output_pano_r, output_pano_l, feature_num=75000)
            output_pano = stitch_two_frames(output_pano, cur_pano, feature_num=100000)
        visualize_output(output_pano)
        
    
    return 
        
if __name__ == "__main__":
    main()