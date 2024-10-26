import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display
from PIL import Image
import easyocr
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#import pytesseract as pt

model = tf.keras.models.load_model("C:\\Users\\aditi\\number_plate_detection.keras")


def object_detection(path, filename):
    # Read image
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image1 = load_img(path, target_size=(224, 224))
    # Data preprocessing
    # Convert into array and get the normalized output
    image_arr_224 = img_to_array(image1)/255.0
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    # Make predictions
    coords = model.predict(test_arr)
    # Denormalize the values
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # Draw bounding on top the image
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    # Convert into bgr
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
    return coords

def save_text(filename, text):
    name, ext = os.path.splitext(filename)
    with open('./static/predict/{}.txt'.format(name), mode='w') as f:
        f.write(text)
    f.close()


def OCR(path, filename):
    img = np.array(load_img(path))
    cods = object_detection(path, filename)
    xmin, xmax, ymin, ymax = cods[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
    
    # Display the ROI image
    display(Image.fromarray(roi_bgr))  # Display the ROI in the notebook
    
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])  # Specify the languages you want to read

    # Extract text from the cropped image
    detections = reader.readtext(magic_color)  # Use the processed image for text extraction

    # Dictionary for mapping license plate prefixes to state names
    states_mapping = {
        "AN": "Andaman and Nicobar Islands",
        "AP": "Andhra Pradesh",
        "AR": "Arunachal Pradesh",
        "AS": "Assam",
        "BR": "Bihar",
        "CG": "Chhattisgarh",
        "CH": "Chandigarh",
        "DD": "Daman and Diu",
        "DL": "Delhi",
        "DN": "Dadra and Nagar Haveli",
        "GA": "Goa",
        "GJ": "Gujarat",
        "HR": "Haryana",
        "HP": "Himachal Pradesh",
        "JH": "Jharkhand",
        "JK": "Jammu and Kashmir",
        "KA": "Karnataka",
        "KL": "Kerala",
        "LD": "Lakshadweep",
        "MH": "Maharashtra",
        "ML": "Meghalaya",
        "MN": "Manipur",
        "MP": "Madhya Pradesh",
        "MZ": "Mizoram",
        "NL": "Nagaland",
        "OD": "Odisha",
        "PB": "Punjab",
        "PY": "Puducherry",
        "RJ": "Rajasthan",
        "SK": "Sikkim",
        "TN": "Tamil Nadu",
        "TR": "Tripura",
        "TS": "Telangana",
        "UK": "Uttarakhand",
        "UP": "Uttar Pradesh",
        "WB": "West Bengal",
        "LA": "Ladakh"
    }

    # Collect the extracted text
    text_output = []
    state = "Unknown"

    for detection in detections:
        plate_number = detection[1]  # Extract the detected license plate text
        text_output.append(plate_number)  # Append detected text to the list
        
        # Determine the state from the license plate prefix (e.g., first 2 characters)
        if len(plate_number) >= 2:
            prefix = plate_number[:2].upper()  # Get the first two characters and make uppercase
            state = states_mapping.get(prefix, "Unknown")  # Look up the state using the prefix

    # Print the extracted plate number and the determined state
    #print(f"Extracted Plate: {plate_number}")
    #print(f"The car belongs to: {state}")

    # Save the plate number as needed
    save_text(filename, plate_number)  # Save the extracted plate number

    # Return both plate number and state as individual values
    return plate_number, state  # Return both the plate number and the detected state


    # # Print the extracted text
    # for text in text_output:
    #     print(text)  # Print each detected text
    
    # # Print the determined state
    # print(f"The car belongs to: {state}")

    # # You may want to save the text as needed
    # save_text(filename, "\n".join(text_output))  # Save the extracted text
    # return text_output, state  # Return the extracted text as a list


def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf