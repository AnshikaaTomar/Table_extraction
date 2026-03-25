import cv2
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
from img_preprocessor import preprocess

def detect_and_extract_tables(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 2.0
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    table_grid = cv2.add(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables_text = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 100 or h < 100:
            continue

        table_region = thresh[y:y+h, x:x+w]
        img_region = img[y:y+h, x:x+w]
        cell_contours, _ = cv2.findContours(table_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for cc in cell_contours:
            cx, cy, cw, ch = cv2.boundingRect(cc)
            if cw > 20 and ch > 10 and cw < w * 0.98 and ch < h * 0.98:
                cells.append((cx, cy, cw, ch))

        if not cells:
            continue

        cells.sort(key=lambda c: (c[1], c[0]))
        rows = []
        current_row = [cells[0]]
        for cell in cells[1:]:
            if abs(cell[1] - current_row[-1][1]) < 15:
                current_row.append(cell)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cell]
        rows.append(sorted(current_row, key=lambda c: c[0]))

        table_data = []
        for row in rows:
            row_data = []
            for (cx, cy, cw, ch) in row:
                pad = 3
                cell_img = img_region[max(0, cy+pad):cy+ch-pad, max(0, cx+pad):cx+cw-pad]
                if cell_img.size == 0:
                    row_data.append("")
                    continue
                cell_pil = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
                cell_processed = preprocess(cell_pil)
                config = r"--psm 7 --oem 3"
                cell_text = pytesseract.image_to_string(cell_processed, lang="eng+hin", config=config).strip()
                row_data.append(cell_text)
            table_data.append(row_data)

        if table_data:
            df = pd.DataFrame(table_data)
            tables_text.append(df.to_string(index=False, header=False))

    return tables_text