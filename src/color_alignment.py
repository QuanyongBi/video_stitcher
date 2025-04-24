import cv2
import numpy as np
from scipy.interpolate import interp1d


def correct_all_frames(frames, ref_idx=None):
    """
    Apply global luminance and color correction to all frames using a reference frame.
    
    Args:
        frames: List of input frames
        ref_idx: Index of the reference frame. If None, the middle frame is used.
        
    Returns:
        List of corrected frames
    """
    if len(frames) < 2:
        return frames
    
    # Use middle frame as reference if not specified
    if ref_idx is None:
        ref_idx = len(frames) // 2
    
    reference_frame = frames[ref_idx]
    corrected_frames = []
    
    for i, frame in enumerate(frames):
        if i == ref_idx:
            # Skip reference frame
            corrected_frames.append(frame)
        else:
            # Apply both luminance and color correction
            corrected = correct_luminance_and_color(frame, reference_frame)
            corrected_frames.append(corrected)
            
        print(f"Color corrected frame {i+1}/{len(frames)}")
    
    return corrected_frames


def correct_luminance_and_color(target, reference):
    """
    Apply luminance and color correction to the target frame based on the reference frame.
    
    Args:
        target: Target frame to be corrected
        reference: Reference frame
        
    Returns:
        Corrected frame
    """
    # Convert to LAB color space
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l_target, a_target, b_target = cv2.split(target_lab)
    l_reference, a_reference, b_reference = cv2.split(reference_lab)
    
    # Correct luminance using histogram matching
    l_corrected = match_histograms(l_target, l_reference)
    
    # # Correct color channels
    a_corrected = correct_color_channel(a_target, a_reference)
    b_corrected = correct_color_channel(b_target, b_reference)
    
    # Merge channels
    corrected_lab = cv2.merge([l_corrected, a_corrected, b_corrected])
    
    # Convert back to BGR
    corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    return corrected


def match_histograms(source, reference):
    """
    Match the histogram of the source to the reference.
    
    Args:
        source: Source image channel
        reference: Reference image channel
        
    Returns:
        Histogram-matched source channel
    """
    # Calculate histograms
    hist_source, bins = np.histogram(source.flatten(), 256, [0, 256])
    hist_reference, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution functions (CDFs)
    cdf_source = hist_source.cumsum()
    cdf_source = cdf_source / float(cdf_source.max())
    
    cdf_reference = hist_reference.cumsum()
    cdf_reference = cdf_reference / float(cdf_reference.max())
    
    # Create lookup table
    lookup_table = np.zeros(256)
    
    # Find the closest CDF value
    for i in range(256):
        if cdf_source[i] != 0:
            # Find the closest match in the reference CDF
            idx = np.abs(cdf_reference - cdf_source[i]).argmin()
            lookup_table[i] = idx
    
    # Apply lookup table
    matched = cv2.LUT(source, lookup_table.astype('uint8'))
    
    return matched


def correct_color_channel(source, reference):
    """
    Correct a color channel (a or b in LAB) based on the reference channel.
    
    Args:
        source: Source color channel
        reference: Reference color channel
        
    Returns:
        Corrected color channel
    """
    # Calculate statistics
    mean_source = np.mean(source)
    std_source = np.std(source)
    
    mean_reference = np.mean(reference)
    std_reference = np.std(reference)
    
    # Apply mean and standard deviation correction with a weight factor
    # to avoid over-saturation
    weight = 0.8  # Adjust between 0 and 1
    
    # Normalize
    normalized = (source - mean_source)
    
    # Scale by the ratio of standard deviations
    if std_source > 0:
        normalized = normalized * (std_reference / std_source)
    
    # Apply weighted correction - blend between original and fully corrected
    corrected = (normalized + mean_reference) * weight + source * (1 - weight)
    
    # Clip to valid range for a,b channels in LAB (-128 to 127)
    corrected = np.clip(corrected, 0, 255).astype('uint8')
    
    return corrected