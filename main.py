import os
import numpy as np
from PIL import Image
from config import input_dir, output_dir, supported_formats
from img_preprocessor import preprocess, extract_text
from clean_text import clean_text
from file_handling import load_file, crop_header
from extract import detect_and_extract_tables


def process_file(filepath):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    print(f"\nProcessing: {filename}")

    pages = load_file(filepath)
    if not pages:
        return

    all_text = []

    for i, pil_image in enumerate(pages):
        print(f"  → Page {i + 1}/{len(pages)}")

        if i == 0:
            pil_image = crop_header(pil_image)

        tables, table_mask = detect_and_extract_tables(pil_image)
        # Mask out table regions before regular OCR
        if table_mask is not None:
            img_arr = np.array(pil_image)
            img_arr[table_mask > 0] = 255  # white out table area
            pil_image = Image.fromarray(img_arr)

        processed = preprocess(pil_image)
        text = extract_text(processed)
        text = clean_text(text)

        page_content = text
        if tables:
            page_content += "\n\n--- TABLES ---\n\n"
            page_content += "\n\n".join(tables)

        all_text.append(f"--- Page {i + 1} ---\n{page_content}")

    final_text = "\n\n".join(all_text)
    output_path = os.path.join(output_dir, f"{name_without_ext}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"  ✓ Saved: {output_path}")


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    all_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]

    if not all_files:
        print("No supported files found in inputs folder.")
    else:
        print(f"Found {len(all_files)} file(s) to process.")
        for filepath in all_files:
            process_file(filepath)