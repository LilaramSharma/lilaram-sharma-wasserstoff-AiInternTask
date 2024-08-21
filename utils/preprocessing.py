from PIL import Image

def resize_image(image_path, size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image

def convert_to_grayscale(image_path):
    image = Image.open(image_path).convert("L")
    return image
