import torch
import torchvision
from torchvision import models, transforms
from PIL import Image

def load_identification_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def identify_objects(image_path):
    # Load the model
    model = load_identification_model()

    # Load and convert the image to RGB
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert the image to a tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    
    # Ensure tensor is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    # Get predictions
    with torch.no_grad():  # Ensure no gradients are calculated
        predictions = model([image_tensor])[0]
    
    return predictions

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def describe_objects(predictions):
    labels = predictions['labels'].tolist()  # Convert tensor to list
    descriptions = []
    
    for label in labels:
        if label < len(COCO_INSTANCE_CATEGORY_NAMES):
            descriptions.append(COCO_INSTANCE_CATEGORY_NAMES[label])
        else:
            descriptions.append("Unknown")
    
    return descriptions

if __name__ == "__main__":
    predictions = identify_objects('data/segmented_objects/object_0.png')
    descriptions = describe_objects(predictions)
    print(descriptions)
