import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np
import os

def load_model():
    # Load the Mask R-CNN model with the most up-to-date weights
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model

def segment_image(image_path):
    model = load_model()
    image = Image.open(image_path).convert("RGB")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor = transform(image)
    
    # Get predictions from the model
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    # Debugging: Print the type and content of predictions
    print("Type of predictions:", type(predictions))
    print("Content of predictions:", predictions)
    
    return predictions

def save_segmented_objects(predictions, image_path, output_dir="data/segmented_objects"):
    image = Image.open(image_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each box in the predictions and save the cropped image
    for i, box in enumerate(predictions['boxes']):
        box = box.detach().numpy()  # Detach and convert to NumPy array
        label = predictions['labels'][i].item()  # Convert label to Python int
        
        # Crop and save the segmented object
        segment = image.crop(box)
        segment.save(os.path.join(output_dir, f"object_{i}.png"))

if __name__ == "__main__":
    predictions = segment_image('data/input_images/sample_image.jpg')
    save_segmented_objects(predictions, 'data/input_images/sample_image.jpg')
