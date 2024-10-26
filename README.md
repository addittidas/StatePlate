# StatePlate

StatePlate is a web application that processes images to detect and display license plate numbers with the state it belongs to. Users can upload an image, and the application will use computer vision techniques to extract and output the license plate number with the respective state.

## Objective
The objective of this project is to develop a comprehensive pipeline for processing raw image data to derive meaningful insights using machine learning (ML) and image processing techniques. This pipeline will focus on the key tasks of data pre-processing, segmentation, detection, and classification, applied in a sequential manner to facilitate effective data analysis and interpretation. The Python programming language, along with essential libraries for image processing and ML, will be used to implement the pipeline.

## Pipeline Overview
The ML and image processing pipeline consists of four main stages:
1.	Data Pre-processing
2.	Segmentation
3.	Object Detection
4.	Classification
Each stage is designed to sequentially transform the raw image data into a structured format that enables accurate interpretation and insight extraction.

## Features

- **Upload Image**: Users can upload images of vehicles with visible license plates.
- **Automatic Plate Detection**: The app detects and extracts the license plate number from the uploaded image and accordingly tells the state the car belongs to.
- **User-Friendly Interface**: Simple, easy-to-use web interface.

## Technologies Used

- **Python**
- **OpenCV**: For image processing and plate detection.
- **Tesseract OCR**: For extracting text from license plates.
- **Flask**: Backend routing and handling user requests.
- **HTML/CSS/Bootstrap**: Frontend layout and styling for user interface.
- **JavaScript**: Client-side interactivity
- **Jupyter Notebook**: For development and testing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/addittidas/StatePlate.git
   cd StatePlate
