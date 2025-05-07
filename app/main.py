from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
import base64
import cv2
import numpy as np
from PIL import Image
import pytesseract
from skimage.metrics import structural_similarity as ssim
from fastapi.middleware.cors import CORSMiddleware


from app.models.scanning import process_document
from app.utils.ocr import extract_text
from app.utils.signature import verify_signature

app = FastAPI(title="Document Processing API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Create upload directory
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Document Processing API"}

@app.post("/scan-document/")
async def scan_document(file: UploadFile = File(...)):
    try:
        # Create temp file
        temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}.jpg")
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process document
        scanned_image = process_document(temp_file_path)
        
        # Read the processed image and encode as base64
        with open(scanned_image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Clean up
        os.remove(temp_file_path)
        os.remove(scanned_image)
        
        return JSONResponse({
            "message": "Document scanned successfully",
            "image": encoded_image
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'scanned_image' in locals() and os.path.exists(scanned_image):
            os.remove(scanned_image)

@app.post("/extract-text/")
async def perform_ocr(file: UploadFile = File(...)):
    try:
        # Process directly without saving
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Convert to PIL Image for pytesseract
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(pil_image)
        
        return JSONResponse({
            "message": "Text extracted successfully",
            "text": text
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-signature/")
async def check_signature(
    document: UploadFile = File(...),
    signature: UploadFile = File(...),
    reference_signature: UploadFile = File(None)
):
    try:
        # Process images directly
        doc_image = cv2.imdecode(
            np.frombuffer(await document.read(), np.uint8), 
            cv2.IMREAD_GRAYSCALE
        )
        sig_image = cv2.imdecode(
            np.frombuffer(await signature.read(), np.uint8), 
            cv2.IMREAD_GRAYSCALE
        )
        
        ref_sig_image = None
        if reference_signature:
            ref_sig_image = cv2.imdecode(
                np.frombuffer(await reference_signature.read(), np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
        
        # Verify signature
        if ref_sig_image is None:
            result = cv2.matchTemplate(doc_image, sig_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            verified = max_val > 0.7
            confidence = float(max_val)
        else:
            sig_image = cv2.resize(sig_image, (200, 100))
            ref_sig_image = cv2.resize(ref_sig_image, (200, 100))
            ssim_score = ssim(sig_image, ref_sig_image)
            verified = ssim_score > 0.6
            confidence = float(ssim_score)
            
        return JSONResponse({
            "message": "Signature verification complete",
            "is_verified": verified,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))