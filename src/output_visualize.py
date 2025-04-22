import cv2
from matplotlib import pyplot as plt

def visualize_output(frames, output):
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax.set_title('Stitched Panorama')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()