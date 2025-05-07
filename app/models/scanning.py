import cv2
import numpy as np
from skimage.filters import threshold_local
import os

def process_document(image_path):
    """
    Process a document image to create a scanned effect:
    1. Detect document edges
    2. Apply perspective transform
    3. Apply thresholding for clean black and white effect
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        str: Path to the processed image
    """
    # Read the image
    image = cv2.imread(image_path)
    orig = image.copy()
    
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Find the document contour
    screen_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            screen_contour = approx
            break
    
    # Handle case where no document contour is found
    if screen_contour is None:
        # Just apply simple preprocessing if no contour found
        warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        warped = (warped > T).astype("uint8") * 255
    else:
        # Apply perspective transform
        warped = four_point_transform(orig, screen_contour.reshape(4, 2))
        
        # Convert to grayscale and threshold
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        warped = (warped > T).astype("uint8") * 255
    
    # Save the processed image
    output_path = f"processed_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, warped)
    
    return output_path

def order_points(pts):
    """
    Order the points in a quadrilateral: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts (numpy.ndarray): The four points to order
        
    Returns:
        numpy.ndarray: Ordered points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of (x,y) coordinates
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    
    # Difference of (x,y) coordinates
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
    
    return rect

def four_point_transform(image, pts):
    """
    Apply a perspective transform to an image based on four points
    
    Args:
        image (numpy.ndarray): The input image
        pts (numpy.ndarray): Four points defining the region to transform
        
    Returns:
        numpy.ndarray: The warped image
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))
    
    # Calculate height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(heightA), int(heightB))
    
    # Define the destination points for the transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    # Calculate the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped