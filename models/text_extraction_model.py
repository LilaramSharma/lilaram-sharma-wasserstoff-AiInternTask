import easyocr

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    text = " ".join([res[1] for res in result])
    return text

if __name__ == "__main__":
    text = extract_text_from_image('data/segmented_objects/object_0.png')
    print(text)
