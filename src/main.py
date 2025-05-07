import argparse
import os
import sys
import time
import cv2
from datetime import datetime

from video_utils import extract_and_save_frames
from video_stitcher_test import (
    stitch_images_stack, 
    stitch_two_frames
)
from color_alignment import correct_all_frames, match_histograms, correct_all_videos
from output_visualize import visualize_output


def get_video_files(directory):
    """Get all video files from the specified directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for file in os.listdir(directory):
        ext = os.path.splitext(file)[1].lower()
        if ext in video_extensions:
            video_files.append(os.path.join(directory, file))
    return sorted(video_files)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Video Stitcher - Create ultra-high-resolution images from videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py --video-dir ./data/video_data/video1 --interval 5 --pre-operation color
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--video-dir', 
        type=str, 
        required=True,
        help='Directory containing video files to process'
    )
    
    # Optional arguments
    parser.add_argument(
        '--interval', 
        type=int, 
        default=10,
        help='Interval for frame extraction (default: 10)'
    )
    
    parser.add_argument(
        '--pre-operation', 
        type=str, 
        choices=['none', 'color', 'histogram',],
        default='none',
        help='Pre-operation to apply to frames (default: none)'
    )
    
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['stack', 'opencv'],
        default='stack',
        help='Stitching method to use (default: stack)'
    )
    
    return parser.parse_args()
    
def perform_pre_operations(frames_per_file, operation):
    """Apply pre-operations to frames based on user choice."""
    # Default using first video's middle frame as reference frame
    ref_frame = frames_per_file[0][len(frames_per_file[0]) // 2]
    
    if operation == 'none':
        print("No pre-operations applied.")
        return frames_per_file
    
    elif operation == 'color':
        print("Applying color correction...")
        return correct_all_videos(frames_per_file, ref_frame)
    
    # TODO: Error here
    elif operation == 'histogram':
        print("Applying Histogram Matching...")
        return match_histograms(frames_per_file, ref_frame)
    
    return frames_per_file

def stitch_frames(frames, corrected_frames, method='stack', debug=False):
    
    start_time = time.time()
    
    if method == 'linear':
        print('not supported')
    elif method == 'stack':
        pano_l, pano_l_corrected = stitch_images_stack(frames[0:len(frames) // 2], corrected_frames[0:len(frames) // 2])
        pano_r, pano_r_corrected = stitch_images_stack(frames[(len(frames) // 2 - 1):len(frames)], corrected_frames[(len(frames) // 2 - 1):len(frames)])
        output_pano, output_corrected = stitch_two_frames(pano_r, pano_r_corrected, pano_l, pano_l_corrected, feature_num=75000)
    elif method == 'divide-conquer':
        print('not supported')
    else:
        raise ValueError(f"Unknown stitching method: {method}")
    
    elapsed_time = time.time() - start_time
    print(f"Stitching completed in {elapsed_time:.2f} seconds")
    
    return output_pano, output_corrected

def main():
    args = parse_arguments()
    
    print("=" * 50)
    print("Video Stitcher - Configuration")
    print("=" * 50)
    print(f"Video directory: {args.video_dir}")
    print(f"Frame interval: {args.interval}")
    print(f"Pre-operation: {args.pre_operation}")
    print(f"Stitching method: {args.method}")
    print("=" * 50)
    
    # Check if video directory exists
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory '{args.video_dir}' does not exist")
        sys.exit(1)
    
    # Get video files
    video_files = get_video_files(args.video_dir)
    if not video_files:
        print(f"Error: No video files found in '{args.video_dir}'")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video file(s):")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video_file)}")
    print()
    
    # Extract frames from videos
    # Each video file would be proceeded seperatly
    print("Extracting frames from videos...")
    frames_count = 0
    frames_per_file = []
    for video in video_files: 
        frames = extract_and_save_frames(
            video, 
            interval=args.interval
        )
        # Apply pre-operations
        frames_per_file.append(frames)
        frames_count += len(frames)
    print(f"Successfully extracted {frames_count} frames")
        
    frames_per_file_corrected = perform_pre_operations(
        frames_per_file, 
        args.pre_operation, 
    )
    
    if len(frames_per_file) == 0:
        print("Error: No frames extracted from videos")
        sys.exit(1)
    
    
    # Stitch each video as an independent pano image
    stiched_pano = []
    stiched_pano_corrected = []
    for i in range(len(frames_per_file)):
        video_frames = frames_per_file[i]
        corrected_video_frames = frames_per_file_corrected[i]
        pano, pano_corrected = stitch_frames(video_frames, corrected_video_frames, method=args.method)
        # visualize_output(pano)
        # visualize_output(pano_corrected)
        stiched_pano.append(pano)
        stiched_pano_corrected.append(pano_corrected)
    # Then stitch each video's pano together into a big pano
    output, _ = stitch_images_stack(stiched_pano, stiched_pano_corrected)
    
    if output is None:
        print("Error: Stitching failed")
        sys.exit(1)
    
    # Visualize output if not disabled
    print("Displaying stitched panorama...")
    visualize_output(output)
    
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()