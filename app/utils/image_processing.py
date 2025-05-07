import cv2
import numpy as np

def enhance_image(image):
    """
    Apply basic image enhancement techniques
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def deskew_image(image):
    """
    Deskew an image to straighten text
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Deskewed image
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find all non-zero points
    coords = np.column_stack(np.where(thresh > 0))
    
    # Find the minimum area rectangle that encloses the text
    rect = cv2.minAreaRect(coords)
    angle = rect[2]
    
    # Adjust the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def binarize_image(image, adaptive=True):
    """
    Convert image to binary (black and white)
    
    Args:
        image (numpy.ndarray): Input image
        adaptive (bool): Whether to use adaptive thresholding
        
    Returns:
        numpy.ndarray: Binarized image
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if adaptive:
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2)
    else:
        # Global thresholding
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    return binary