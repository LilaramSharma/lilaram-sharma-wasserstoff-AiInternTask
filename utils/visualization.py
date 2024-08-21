import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_segmented_objects(image_path, predictions):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for j, box in enumerate(predictions['boxes']):
        box = box.detach().numpy()  # Fix: Detach tensor and convert to NumPy array
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
