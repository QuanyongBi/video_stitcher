import argparse
import os
import sys
import time
import cv2
from datetime import datetime

from video_utils import extract_and_save_frames
from video_stitcher import (
    stitch_images_linear, 
    stitch_images_stack, 
    stitch_images_divide_conquer
)
from color_alignment import correct_all_frames, match_histograms
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
    
def perform_pre_operations(frames, operation):
    """Apply pre-operations to frames based on user choice."""
    if operation == 'none':
        print("No pre-operations applied.")
        return frames
    
    elif operation == 'color':
        print("Applying color correction...")
        return correct_all_frames(frames)
    
    # TODO: Error here
    elif operation == 'histogram':
        print("Applying Histogram Matching...")
        return match_histograms(frames)
    
    return frames

def stitch_frames(frames, method='stack', debug=False):
    print(f"Stitching {len(frames)} frames using {method} method...")
    
    start_time = time.time()
    
    if method == 'linear':
        output = stitch_images_linear(frames, debug_progress=debug)
    elif method == 'stack':
        output = stitch_images_stack(frames)
    elif method == 'divide-conquer':
        output = stitch_images_divide_conquer(frames)
    else:
        raise ValueError(f"Unknown stitching method: {method}")
    
    elapsed_time = time.time() - start_time
    print(f"Stitching completed in {elapsed_time:.2f} seconds")
    
    return output

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
    print("Extracting frames from videos...")
    frames = extract_and_save_frames(
        video_files, 
        output_dir="data/extracted_frames",
        interval=args.interval
    )
    
    if not frames:
        print("Error: No frames extracted from videos")
        sys.exit(1)
    
    print(f"Successfully extracted {len(frames)} frames")
    
    # Apply pre-operations
    frames = perform_pre_operations(
        frames, 
        args.pre_operation, 
    )
    
    # Stitch frames
    output = stitch_frames(frames, method=args.method)
    
    if output is None:
        print("Error: Stitching failed")
        sys.exit(1)
    
    # Visualize output if not disabled
    print("Displaying stitched panorama...")
    visualize_output(output)
    
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()