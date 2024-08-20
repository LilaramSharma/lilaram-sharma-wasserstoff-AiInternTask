# lilaram-sharma-wasserstoff-AiInternTask

Project Overview

This project is designed to build a robust pipeline that processes images by segmenting, identifying, and analyzing objects within them. Using state-of-the-art deep learning models, the pipeline automatically segments all visible objects in an image, extracts each object separately, and identifies them with detailed descriptions. The project then leverages OCR techniques to extract any text or data from the objects, followed by generating a comprehensive summary of their attributes. Finally, all extracted and analyzed data is mapped back to the original image, resulting in a final output that includes the annotated image and a summary table with information for each object. The pipeline is modular, allowing for easy adjustments and enhancements, and is packaged with a Streamlit-based user interface for testing and visualization.

Setup Instructions

To set up the project, start by cloning the repository to your local machine and installing the required dependencies. This project is built with Python, and it leverages libraries such as PyTorch, TensorFlow, and OpenCV. Once the dependencies are installed, you can run the pipeline through a Streamlit application, which provides an interactive interface for uploading images, processing them, and viewing the results. The input images should be placed in the data/input_images/ directory, and the results, including segmented objects and final outputs, will be saved in designated subdirectories within the data/ folder.

Usage Guidelines

The pipeline is structured to guide users through a series of steps, starting with the segmentation of objects in an image. Users can then extract these objects, store them with unique identifiers, and use pre-trained models to identify and describe each one. The pipeline also includes functionality to extract any embedded text or data from the objects and summarizes the findings in an easily interpretable format. Finally, the entire process is mapped back to the original image, producing a final annotated image and a summary table that encapsulates all relevant data. Users can interact with each step of the pipeline through the Streamlit app, making it easy to visualize and review the results.
