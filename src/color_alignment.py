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


def correct_color_channel(source, reference):
    """
    Correct a color channel (a or b in LAB) based on the reference channel.
    """
    # Calculate statistics
    mean_source = np.mean(source)
    std_source = np.std(source)
    
    mean_reference = np.mean(reference)
    std_reference = np.std(reference)
    
    # Determine appropriate weight based on histogram similarity
    hist_source, _ = np.histogram(source.flatten(), 256, [0, 256])
    hist_reference, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    # Normalize histograms for comparison
    hist_source = hist_source / np.sum(hist_source)
    hist_reference = hist_reference / np.sum(hist_reference)
    
    # Calculate histogram intersection (higher means more similar)
    similarity = np.sum(np.minimum(hist_source, hist_reference))
    
    # Adaptive weight: more different = stronger correction
    weight = max(0.5, min(0.9, 1.0 - similarity))
    
    # Apply mean and standard deviation correction
    normalized = (source.astype(np.float32) - mean_source)
    
    # Scale by the ratio of standard deviations (avoid division by zero)
    if std_source > 1e-5:
        normalized = normalized * (std_reference / std_source)
    
    # Apply weighted correction
    corrected = (normalized + mean_reference) * weight + source * (1 - weight)
    
    # Clip to valid OpenCV LAB range for a,b channels (usually 0-255 in OpenCV)
    # This will be correctly mapped to -128 to 127 range internally
    corrected = np.clip(corrected, 0, 255).astype('uint8')
    
    return corrected

def match_histograms(source, reference):
    """
    Match the histogram of source to reference using a more robust approach.
    """
    # Calculate CDFs
    hist_source, bins = np.histogram(source.flatten(), 256, [0, 256])
    hist_reference, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    # Calculate normalized cumulative histograms
    cdf_source = np.cumsum(hist_source).astype(np.float64)
    cdf_source /= cdf_source[-1]
    
    cdf_reference = np.cumsum(hist_reference).astype(np.float64)
    cdf_reference /= cdf_reference[-1]
    
    # Create interpolation function for smoother mapping
    # This creates a monotonically increasing mapping
    interp_fn = interp1d(
        np.arange(256), 
        cdf_source,
        bounds_error=False, 
        fill_value=(0, 1)
    )
    source_values = interp_fn(np.arange(256))
    
    # Find the corresponding reference values
    lookup_table = np.zeros(256)
    for i in range(256):
        # Find reference level with the closest CDF value
        if source_values[i] <= 0:
            lookup_table[i] = 0
        elif source_values[i] >= 1:
            lookup_table[i] = 255
        else:
            # Find where in the reference CDF we get the same value
            lookup_table[i] = np.interp(source_values[i], cdf_reference, np.arange(256))
    
    # Apply lookup table
    matched = cv2.LUT(source, lookup_table.astype('uint8'))
    
    return matched