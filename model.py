import cv2
from ultralytics import YOLO
import numpy as np
import re
import os
from paddleocr import PaddleOCR
import csv
import time
import imutils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Đường dẫn mặc định
VIDEO_PATH = "http://192.168.100.7:8080/video"  # Camera
OUTPUT_CSV = "output.csv"

# Configuration parameters for optimization
OCR_CONFIDENCE_THRESHOLD = 80  # Minimum confidence for OCR (0-100)

# Khởi tạo mô hình
model = YOLO("runs/detect/train/weights/best.pt")
ocr = PaddleOCR(
    use_angle_cls=True, 
    use_gpu=True,
    rec_model_dir=r"D:\PaddleOCR\inference\license_plate_rec",
    # If you used a custom dictionary during training, specify it here:
    rec_char_dict_path=r"D:\PaddleOCR\ppocr\utils\license_plate_dict.txt",
    # Set det=False since you're only interested in text recognition
    det=False
)

# ---------- License Plate Flattening Functions ----------
def order_points(pts):
    """
    Order points in a specific order (top-left, top-right, bottom-right, bottom-left)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    """
    Apply a perspective transform to flatten a plate
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def find_and_flatten_plate(plate_img):
    """
    Find contours and flatten a license plate image
    Returns the flattened plate or the original if flattening fails
    """
    try:
        # Make a copy to avoid modifying the original
        original = plate_img.copy()
        
        # Resize for consistent processing
        ratio = plate_img.shape[0] / 300.0
        plate_img = imutils.resize(plate_img, height=300)
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        
        # Threshold to get binary image
        thresh = cv2.threshold(cl1, 165, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
        
        # Find the largest contour that meets our criteria
        for c in cnts:
            # Skip very small contours
            if cv2.contourArea(c) < 5000:  
                continue
            
            # Find minimum area rectangle around the contour
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.intp)
            
            # Apply four-point perspective transform
            warped = four_point_transform(plate_img, box)
            return warped
        
        # If no suitable contour found, return the original
        return original
    except Exception as e:
        print(f"Error in flattening plate: {e}")
        # Return original image if any error occurs
        return plate_img
# ---------------------------------------------------------

def detect_plate_type(plate_img):
    """
    Determine if the license plate is rectangle or square based on aspect ratio
    Now using the flattened plate image
    """
    height, width = plate_img.shape[:2]
    return "rectangle" if width / height > 3 else "square"

def split_license_plate(plate_img):
    """
    Split the flattened license plate into two lines for square plates
    """
    height, width = plate_img.shape[:2]
    line1 = plate_img[0:height // 2, :]
    line2 = plate_img[height // 2:, :]
    return line1, line2

def preprocess_image(img):
    """Preprocess the image for OCR"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def paddle_ocr_line(img):
    processed_img = preprocess_image(img)
    """Extract text from an image using PaddleOCR"""
    result = ocr.ocr(processed_img, det=False, rec=True, cls=False)
    text = ""
    score = 0
    for r in result:
        curr_score = int(r[0][1] * 100) if not np.isnan(r[0][1]) else 0
        if curr_score > OCR_CONFIDENCE_THRESHOLD:
            text = r[0][0]
            score = curr_score
    return re.sub('[\W]', '', text), score

def process_license_plate(image, x1, y1, x2, y2):
    """Process a detected license plate and extract its text"""
    # Ensure bounding box coordinates are within the frame dimensions
    height, width, _ = image.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    # Extract the license plate region
    plate_region = image[y1:y2, x1:x2]
    
    # Apply flattening to the plate region
    flattened_plate = find_and_flatten_plate(plate_region)
    
    # Determine plate type based on the flattened plate (new logic)
    plate_type = detect_plate_type(flattened_plate)

    # Define dictionaries for character fixes
    number_to_char = {'0': 'D', '1': 'L', '3': 'B', '4': 'A', '6': 'G', '5': 'S', '8': 'B'}
    char_to_number = {'O': '0', 'I': '1', 'J': '1', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'B': '8'}
    
    # Invalid 3rd position letters
    invalid_3rd_chars = {'I', 'J', 'O', 'Q', 'W'}
    # Alternative mappings for invalid 3rd position characters
    third_char_fixes = {'I': 'H', 'J': 'K', 'O': 'D', 'Q': 'D', 'W': 'V'}

    if plate_type == "rectangle":
        if flattened_plate.size == 0:  # Check if ROI is empty
            return "", 0
        text, score = paddle_ocr_line(flattened_plate)
        # Validate character count for rectangle plates
        if len(text) in [7, 8]:
            text = fix_license_plate(text, number_to_char, char_to_number, third_char_fixes, invalid_3rd_chars)
            if validate_license_plate(text, plate_type):
                return text, score
        return "", 0  # Return empty if character count or format is invalid
    
    elif plate_type == "square":
        # Split the flattened plate for square plates
        line1, line2 = split_license_plate(flattened_plate)
        
        if line1.size == 0 or line2.size == 0:  # Check if ROIs are empty
            return "", 0
        text1, score1 = paddle_ocr_line(line1)
        text2, score2 = paddle_ocr_line(line2)
        
        # Store original detected text for format validation
        original_text1 = text1
        
        text = text1 + text2
        score = min(score1, score2)  # Use the minimum confidence score
        
        # Validate character count for square plates
        if len(text) in [7, 8, 9]:
            text = fix_license_plate(text, number_to_char, char_to_number, third_char_fixes, invalid_3rd_chars)
            # Apply additional rule for the 4th character based on first line length
            if len(original_text1) == 4 and len(text) >= 4:
                # If first line has 4 characters: 4th character can be letter or number (no change needed)
                pass
            elif len(original_text1) == 3 and len(text) >= 4:
                # If first line has 3 characters: 4th character must be a number
                if text[3].isalpha():
                    if text[3] in char_to_number:
                        # Convert any letter at 4th position to its corresponding number
                        fourth_char_list = list(text)
                        fourth_char_list[3] = char_to_number[text[3]]
                        text = ''.join(fourth_char_list)
                    else:
                        # If no mapping exists, make it a default number like '0'
                        fourth_char_list = list(text)
                        fourth_char_list[3] = '0'
                        text = ''.join(fourth_char_list)
            
            if validate_license_plate(text, plate_type, original_text1_len=len(original_text1)):
                return text, score
        return "", 0  # Return empty if character count or format is invalid
    
    return "", 0

def fix_license_plate(text, number_to_char, char_to_number, third_char_fixes, invalid_3rd_chars):
    """Fix the license plate text based on the dictionaries and specific format rules"""
    if len(text) < 3:
        return text
    
    fixed_text = list(text)
    
    # Ensure first two characters are digits
    for i in range(2):
        if i < len(fixed_text) and fixed_text[i] in char_to_number:
            fixed_text[i] = char_to_number[fixed_text[i]]
    
    # Fix the 3rd character to be a valid letter
    if len(fixed_text) > 2:
        # If it's a number that can be mapped to a letter
        if fixed_text[2].isdigit() and fixed_text[2] in number_to_char:
            fixed_text[2] = number_to_char[fixed_text[2]]
        
        # If it's an invalid letter, replace with a valid alternative
        if fixed_text[2] in invalid_3rd_chars and fixed_text[2] in third_char_fixes:
            fixed_text[2] = third_char_fixes[fixed_text[2]]
    
    # Fix the rest of the characters to be digits (except for position 4 which has special rules)
    # Note: The special rule for position 4 is handled separately in process_license_plate
    for i in range(3, len(fixed_text)):
        if i != 3 and fixed_text[i] in char_to_number:  # Skip position 4 here
            fixed_text[i] = char_to_number[fixed_text[i]]
    
    return ''.join(fixed_text)

def validate_license_plate(text, plate_type, original_text1_len=None):
    """Validate the license plate format with the new rules"""
    if len(text) < 7:  # Ensure the text has at least 7 characters
        return False
    
    # Check if first two characters form a valid 2-digit number between 11 and 99
    if len(text) < 2 or not text[0:2].isdigit():
        return False
    
    plate_prefix = int(text[0:2])
    if plate_prefix < 11 or plate_prefix > 99:
        return False
    
    # The 3rd character must be a letter in the alphabet, excluding I, J, O, Q and W
    invalid_chars = {'I', 'J', 'O', 'Q', 'W'}
    if not text[2].isalpha() or text[2] in invalid_chars:
        return False
    
    # Apply special rules for square plates
    if plate_type == "square" and original_text1_len is not None and len(text) >= 4:
        # If first line has 3 characters: 4th character must be a number
        if original_text1_len == 3 and not text[3].isdigit():
            return False
        # For first line with 4 characters, 4th character can be either letter or number (no validation needed)
    
    # The rest of the characters (except possibly position 4) must be numbers from 0-9
    for i in range(3, len(text)):
        # Skip position 4 check for square plates where first line has 4 characters
        if plate_type == "square" and original_text1_len == 4 and i == 3:
            continue
        if not text[i].isdigit():
            return False
    
    # Additional validation for plate type
    if plate_type == "rectangle" and len(text) not in [7, 8]:
        return False
    if plate_type == "square" and len(text) not in [7, 8, 9]:
        return False
    
    return True

def process_frame(frame, output_csv=OUTPUT_CSV):
    """Process a frame to detect license plates"""
    # Create a copy of the frame to work with
    process_frame = frame.copy()
    
    # Run detection on the frame
    results = model(process_frame)
    
    found_plates = []
    
    # Process each detected license plate
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        # Draw bounding box
        cv2.rectangle(process_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Process the license plate
        plate_text, ocr_confidence = process_license_plate(process_frame, x1, y1, x2, y2)
        
        if plate_text:
            # Get the plate region for determining plate type
            plate_region = process_frame[y1:y2, x1:x2]
            flattened_plate = find_and_flatten_plate(plate_region)
            plate_type = detect_plate_type(flattened_plate)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to results list
            found_plates.append((plate_text, plate_type, ocr_confidence, timestamp))
            
            # Display the plate text on the image
            cv2.putText(process_frame, f"Plate: {plate_text}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(process_frame, f"Conf: {ocr_confidence}%", (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write to CSV only if output_csv is provided
            if output_csv is not None:
                with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([plate_text, plate_type, ocr_confidence, timestamp])
            
            print(f"Found plate: {plate_text} ({plate_type}) with confidence {ocr_confidence}")
    
    # Display the results
    if found_plates:
        result_text = f"Found {len(found_plates)} license plate(s)"
    else:
        result_text = "No license plates detected"
    
    cv2.putText(process_frame, result_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return process_frame, found_plates

def init_csv(output_csv=OUTPUT_CSV):
    """Initialize the CSV file with headers"""
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Plate Number", "Plate Type", "Confidence", "Timestamp"])

# Chỉ chạy khi file này được gọi trực tiếp
if __name__ == "__main__":
    # Khởi tạo camera
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Khởi tạo file CSV
    init_csv()
    
    # Thiết lập cửa sổ hiển thị
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Feed", 800, 600)
    
    print("Press SPACE to capture and process a frame, or 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Hiển thị video từ camera
        cv2.imshow("Camera Feed", frame)
        
        # Đợi phím nhấn
        key = cv2.waitKey(1) & 0xFF
        
        # Nếu nhấn phím cách (space), xử lý frame hiện tại
        if key == ord(' '):
            print("Processing frame...")
            
            # Hiển thị thông báo đang xử lý
            processing_frame = frame.copy()
            cv2.putText(processing_frame, "Processing...", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Camera Feed", processing_frame)
            cv2.waitKey(1)  # Cập nhật màn hình
            
            # Xử lý khung hình
            processed_frame, found_plates = process_frame(frame)
            
            # Hiển thị kết quả
            cv2.imshow("Camera Feed", processed_frame)
        
        # Nếu nhấn 'q', thoát
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()