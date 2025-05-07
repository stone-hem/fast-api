import pytesseract
from PIL import Image
import cv2
import numpy as np
import os

def extract_text(image_path):
    """
    Extract text from an image using OCR
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        str: Extracted text
    """
    # Preprocess the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Apply some blur to reduce noise
    gray = cv2.medianBlur(gray, 3)
    
    # Save the processed image temporarily
    temp_file = "temp_processed.png"
    cv2.imwrite(temp_file, gray)
    
    # Perform OCR
    text = pytesseract.image_to_string(Image.open(temp_file))
    
    # Clean up
    os.remove(temp_file)
    
    return text

def extract_text_from_memory(image_array):
    """
    Extract text from an image in memory without saving to disk
    
    Args:
        image_array (numpy.ndarray): Image array
        
    Returns:
        str: Extracted text
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    
    # Apply thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Apply some blur to reduce noise
    gray = cv2.medianBlur(gray, 3)
    
    # Convert to PIL Image for pytesseract
    pil_image = Image.fromarray(gray)
    
    # Perform OCR
    text = pytesseract.image_to_string(pil_image)
    
    return text