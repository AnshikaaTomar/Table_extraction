import easyocr
import numpy as np
from PIL import Image 

# Initialize once at module level
ocr_engine = easyocr.Reader(['en'], gpu=True)


def extract_text(processed_image):
    # EasyOCR expects numpy array
    if not isinstance(processed_image, np.ndarray):
        processed_image = np.array(processed_image)

    result = ocr_engine.readtext(
        processed_image,
        detail=0,          
        paragraph=True,    
        confidence_threshold=0.6  
    )

    return "\n".join(result)