import cv2
import numpy as np
import pytesseract
from PIL import Image 
from config import TESSERACT_PATH

# preprocessing
def preprocess(pil_image):
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Convert to grayscale
    scale = 2.0
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # crisp edges for better readability
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, h=20)

    # deskew the image
    coords = np.column_stack(np.where(denoised > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = denoised.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, blockSize= 31, C=15)

    return thresh

# text extraction - OCR
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text(processed_image):
    config = r"--psm 6 --oem 3"
    text = pytesseract.image_to_string(processed_image, lang="eng+hin", config=config)
    return text.strip()