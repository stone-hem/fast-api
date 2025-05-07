import cv2
import numpy as np
from Levenshtein import distance as levenshtein_distance
from skimage.metrics import structural_similarity as ssim

def verify_signature(document_path, signature_path, reference_signature_path=None):
    """
    Verify a signature on a document
    
    Args:
        document_path (str): Path to the document image
        signature_path (str): Path to the signature to verify
        reference_signature_path (str, optional): Path to the reference signature
        
    Returns:
        dict: Verification result with is_verified and confidence
    """
    # Load images
    doc_img = cv2.imread(document_path, cv2.IMREAD_GRAYSCALE)
    sig_img = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)
    
    # Simple verification without reference (just checks if signature is present)
    if reference_signature_path is None:
        # Find signature in document (simple template matching)
        result = cv2.matchTemplate(doc_img, sig_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return {
            "is_verified": max_val > 0.7,  # Threshold
            "confidence": float(max_val)
        }
    else:
        # Compare with reference signature
        ref_img = cv2.imread(reference_signature_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize images to same dimensions
        sig_img = cv2.resize(sig_img, (200, 100))
        ref_img = cv2.resize(ref_img, (200, 100))
        
        # Calculate structural similarity
        ssim_score = ssim(sig_img, ref_img)
        
        return {
            "is_verified": ssim_score > 0.6,  # Threshold
            "confidence": float(ssim_score)
        }

def verify_signature_from_memory(doc_image, sig_image, ref_sig_image=None):
    """
    Verify a signature on a document using in-memory images
    
    Args:
        doc_image (numpy.ndarray): Document image array
        sig_image (numpy.ndarray): Signature image array
        ref_sig_image (numpy.ndarray, optional): Reference signature image array
        
    Returns:
        dict: Verification result with is_verified and confidence
    """
    # Simple verification without reference (just checks if signature is present)
    if ref_sig_image is None:
        # Find signature in document (simple template matching)
        result = cv2.matchTemplate(doc_image, sig_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return {
            "is_verified": max_val > 0.7,  # Threshold
            "confidence": float(max_val)
        }
    else:
        # Resize images to same dimensions
        sig_image = cv2.resize(sig_image, (200, 100))
        ref_sig_image = cv2.resize(ref_sig_image, (200, 100))
        
        # Calculate structural similarity
        ssim_score = ssim(sig_image, ref_sig_image)
        
        return {
            "is_verified": ssim_score > 0.6,  # Threshold
            "confidence": float(ssim_score)
        }