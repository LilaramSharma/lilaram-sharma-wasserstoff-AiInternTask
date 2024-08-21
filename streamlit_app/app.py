import sys
import os
import streamlit as st

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules from models and utils
from models.segmentation_model import segment_image, save_segmented_objects
from models.identification_model import identify_objects, describe_objects
from models.text_extraction_model import extract_text_from_image
from models.summarization_model import summarize_text
from utils.visualization import visualize_segmented_objects

# Streamlit title
st.title("Welcome to Image Segmentation and Object Analysis")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file to a specific directory
    with open(f"data/input_images/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(f"data/input_images/{uploaded_file.name}", caption="Uploaded Image", use_column_width=True)

    # Segment the image
    predictions = segment_image(f"data/input_images/{uploaded_file.name}")
    save_segmented_objects(predictions, f"data/input_images/{uploaded_file.name}")
    visualize_segmented_objects(f"data/input_images/{uploaded_file.name}", predictions)

    # Identify objects in the segmented images
    descriptions = []
    for i in range(len(predictions['boxes'])):
        obj_path = f"data/segmented_objects/object_{i}.png"
        obj_predictions = identify_objects(obj_path)
        descriptions.append(describe_objects(obj_predictions))
    st.write("Identified Objects:", descriptions)

    # Extract text from each segmented object
    extracted_texts = [extract_text_from_image(f"data/segmented_objects/object_{i}.png") for i in range(len(predictions['boxes']))]
    st.write("Extracted Texts:", extracted_texts)

    # Summarize the extracted texts
    summaries = [summarize_text(text) for text in extracted_texts]
    st.write("Summaries:", summaries)
