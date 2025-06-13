import cv2
from ultralytics import YOLO
import numpy as np
import re
import os
from paddleocr import PaddleOCR
import csv
import time
import imutils
from typing import List, Tuple, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Đường dẫn mặc định
VIDEO_PATH = "http://192.168.100.2:8080/video"  # Camera
OUTPUT_CSV = "output.csv"

# Configuration parameters for optimization
OCR_CONFIDENCE_THRESHOLD = 80  # Minimum confidence for OCR (0-100)

# ---------- License Plate Corner Detection Class ----------
class LicensePlateFinder:
    def __init__(self, model_path: str):
        """
        Initialize the license plate corner detector
        
        Args:
            model_path: Path to the YOLOv8n model trained on license plate corners
        """
        self.model = YOLO(model_path)
        self.class_names = ['bottom_left', 'bottom_right', 'top_left', 'top_right']
        
    def detect_corners(self, image):
        """
        Detect license plate corners in the given image
        
        Args:
            image: Input image
            
        Returns:
            warped_plate: Warped license plate image or None if detection fails
            corners: Dictionary of corner coordinates or None if detection fails
            message: Success or error message
        """
        # 1. Detect corners with YOLO
        results = self.model.predict(image, verbose=True, conf = 0.5)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None, "No corners detected"
        
        boxes = results[0].boxes
        
        # Convert to numpy arrays
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        
        # Create dictionary to hold best detection for each class
        best_detections = {}
        
        # 1. Keep only the detection with highest confidence for each class
        for i in range(len(classes)):
            class_id = classes[i]
            confidence = confidences[i]
            
            if class_id not in best_detections or confidence > best_detections[class_id]['confidence']:
                # Calculate center of bounding box
                x1, y1, x2, y2 = coords[i]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                best_detections[class_id] = {
                    'confidence': confidence,
                    'center': (center_x, center_y),
                    'bbox': coords[i]
                }
        
        # 2. Check for overlapping bounding boxes
        ids_to_remove = set()
        for id1 in best_detections:
            for id2 in best_detections:
                if id1 != id2 and id1 not in ids_to_remove and id2 not in ids_to_remove:
                    bbox1 = best_detections[id1]['bbox']
                    bbox2 = best_detections[id2]['bbox']
                    
                    # Calculate intersection
                    overlap = self._calculate_iou(bbox1, bbox2)
                    
                    if overlap > 0.85:  # More than 85% overlap
                        # Remove the one with lower confidence
                        if best_detections[id1]['confidence'] >= best_detections[id2]['confidence']:
                            ids_to_remove.add(id2)
                        else:
                            ids_to_remove.add(id1)
        
        # Remove overlapping detections
        for id_to_remove in ids_to_remove:
            if id_to_remove in best_detections:
                del best_detections[id_to_remove]
        
        # 3. Check if we have enough corners
        if len(best_detections) < 3:
            return None, None, f"Not enough corners detected (found {len(best_detections)})"
        
        # 4. Extract the center coordinates
        corners = {}
        for class_id, detection in best_detections.items():
            corner_name = self.class_names[class_id]
            corners[corner_name] = detection['center']
        
        # 5. If we have 3 corners, infer the 4th one
        if len(corners) == 3:
            missing_corner = self._find_missing_corner(corners)
            corners[missing_corner] = self._infer_fourth_corner(corners)
        
        # 6. Ensure correct positions of corners
        corners = self._validate_and_fix_corners(corners)
        
        # 7. Apply perspective transform
        warped_plate = self._warp_license_plate(image, corners)
        
        return warped_plate, corners, "Success"
    
    def _calculate_iou(self, box1, box2):
        """Calculate intersection over union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area
    
    def _find_missing_corner(self, corners):
        """Find which corner is missing from the detected corners"""
        for corner_name in self.class_names:
            if corner_name not in corners:
                return corner_name
        return None
    
    def _infer_fourth_corner(self, corners):
        """Infer the position of the fourth corner using vector logic"""
        # First check which corner is missing
        missing = set(self.class_names) - set(corners.keys())
        missing_corner = list(missing)[0]
        
        if missing_corner == 'top_left':
            # Vector from bottom_left to bottom_right + vector from bottom_left to top_right
            # Assuming vector addition based on parallelogram logic
            bl = np.array(corners['bottom_left'])
            br = np.array(corners['bottom_right'])
            tr = np.array(corners['top_right'])
            
            # Vector from bl to tr = vector from bl to br + vector from br to tr
            # So tl = bl + (tr - bl) - (br - bl) = tr - br + bl
            return tuple(tr - (br - bl))
            
        elif missing_corner == 'top_right':
            bl = np.array(corners['bottom_left'])
            br = np.array(corners['bottom_right'])
            tl = np.array(corners['top_left'])
            
            # tr = br + (tl - bl)
            return tuple(br + (tl - bl))
            
        elif missing_corner == 'bottom_left':
            tl = np.array(corners['top_left'])
            tr = np.array(corners['top_right'])
            br = np.array(corners['bottom_right'])
            
            # bl = br + (tl - tr)
            return tuple(br + (tl - tr))
            
        elif missing_corner == 'bottom_right':
            tl = np.array(corners['top_left'])
            tr = np.array(corners['top_right'])
            bl = np.array(corners['bottom_left'])
            
            # br = bl + (tr - tl)
            return tuple(bl + (tr - tl))
            
        return None
    
    def _validate_and_fix_corners(self, corners):
        """
        Validate that corners are in the correct positions relative to each other
        and fix them if necessary based on required positioning:
        - top_left: x < top_right.x, y < bottom_left.y, y < bottom_right.y
        - top_right: x > top_left.x, y < bottom_left.y, y < bottom_right.y
        - bottom_left: x < bottom_right.x, y > top_left.y, y > top_right.y
        - bottom_right: x > bottom_left.x, y > top_left.y, y > top_right.y
        """
        # Convert to numpy arrays for easier manipulation
        tl = np.array(corners['top_left'])
        tr = np.array(corners['top_right'])
        bl = np.array(corners['bottom_left'])
        br = np.array(corners['bottom_right'])
        
        # Fix horizontal relationships
        # Top left should be left of top right
        if tl[0] > tr[0]:
            tl[0], tr[0] = tr[0], tl[0]
            
        # Bottom left should be left of bottom right
        if bl[0] > br[0]:
            bl[0], br[0] = br[0], bl[0]
        
        # Fix vertical relationships
        # Top left should be above bottom left
        if tl[1] > bl[1]:
            tl[1], bl[1] = bl[1], tl[1]
        
        # Top left should be above bottom right
        if tl[1] > br[1]:
            tl[1], br[1] = br[1], tl[1]
        
        # Top right should be above bottom left
        if tr[1] > bl[1]:
            tr[1], bl[1] = bl[1], tr[1]
        
        # Top right should be above bottom right
        if tr[1] > br[1]:
            tr[1], br[1] = br[1], tr[1]
        
        # Update corners dictionary with fixed points
        corners['top_left'] = tuple(tl)
        corners['top_right'] = tuple(tr)
        corners['bottom_left'] = tuple(bl)
        corners['bottom_right'] = tuple(br)
        
        return corners
    
    def _warp_license_plate(self, image, corners):
        """Apply perspective transform to get a frontal view of the license plate"""
        # Get corner points in the right order for perspective transform
        src_pts = np.array([
            corners['top_left'],
            corners['top_right'],
            corners['bottom_right'],
            corners['bottom_left']
        ], dtype=np.float32)
        
        # Calculate width and height for the output image
        width_1 = np.sqrt(((corners['top_right'][0] - corners['top_left'][0]) ** 2) + 
                          ((corners['top_right'][1] - corners['top_left'][1]) ** 2))
        width_2 = np.sqrt(((corners['bottom_right'][0] - corners['bottom_left'][0]) ** 2) + 
                          ((corners['bottom_right'][1] - corners['bottom_left'][1]) ** 2))
        
        height_1 = np.sqrt(((corners['bottom_left'][0] - corners['top_left'][0]) ** 2) + 
                           ((corners['bottom_left'][1] - corners['top_left'][1]) ** 2))
        height_2 = np.sqrt(((corners['bottom_right'][0] - corners['top_right'][0]) ** 2) + 
                           ((corners['bottom_right'][1] - corners['top_right'][1]) ** 2))
        
        width = max(int(width_1), int(width_2))
        height = max(int(height_1), int(height_2))
        
        # Define destination points
        dst_pts = np.array([
            [0, 0],  # top-left
            [width, 0],  # top-right
            [width, height],  # bottom-right
            [0, height]  # bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped

# Khởi tạo mô hình - Chỉ giữ lại corner_detector
corner_detector = LicensePlateFinder("runs4goc/detect/train/weights/best.pt")
ocr = PaddleOCR(
    use_angle_cls=True, 
    use_gpu=True,
    rec_model_dir=r"D:\PaddleOCR\inference\license_plate_rec",
    # If you used a custom dictionary during training, specify it here:
    rec_char_dict_path=r"D:\PaddleOCR\ppocr\utils\license_plate_dict.txt",
    # Set det=False since you're only interested in text recognition
    det=False
)

def find_and_flatten_plate(plate_img):
    """
    Find and flatten a license plate image using corner detection
    Returns the flattened plate or the original if flattening fails
    """
    try:
        # Sử dụng corner detector để phát hiện và làm thẳng biển số
        warped_plate, corners, message = corner_detector.detect_corners(plate_img)
        
        # Nếu không thể làm thẳng, trả về ảnh gốc
        if warped_plate is None:
            print(f"Cannot flatten plate: {message}")
            return plate_img
        
        return warped_plate
    except Exception as e:
        print(f"Error in flattening plate: {e}")
        # Return original image if any error occurs
        return plate_img

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
    """Process a frame directly with corner detection"""
    # Create a copy of the frame to work with
    process_frame = frame.copy()
    
    # 1. Chụp frame - đã được thực hiện khi gọi hàm này
    
    # 2. Áp dụng trực tiếp corner detection và transform - chỉ gọi một lần
    flattened_plate = find_and_flatten_plate(process_frame)
    
    # Nếu không phát hiện được 4 góc, hiển thị thông báo và trả về frame gốc
    if flattened_plate is process_frame:  # Trường hợp không thay đổi
        cv2.putText(process_frame, "No license plate corners detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return process_frame, []
    
    # Xác định loại biển số từ biển số đã làm thẳng
    plate_type = detect_plate_type(flattened_plate)
    
    # 3. OCR trực tiếp trên biển số đã làm thẳng
    # Define dictionaries for character fixes
    number_to_char = {'0': 'D', '1': 'L', '3': 'B', '4': 'A', '6': 'G', '5': 'S', '8': 'B'}
    char_to_number = {'O': '0', 'I': '1', 'J': '1', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'B': '8'}
    
    # Invalid 3rd position letters
    invalid_3rd_chars = {'I', 'J', 'O', 'Q', 'W'}
    # Alternative mappings for invalid 3rd position characters
    third_char_fixes = {'I': 'H', 'J': 'K', 'O': 'D', 'Q': 'D', 'W': 'V'}
    
    plate_text = ""
    ocr_confidence = 0
    original_text1 = ""  # Thêm biến để lưu text dòng 1 gốc
    original_text2 = ""  # Thêm biến để lưu text dòng 2 gốc
    
    if plate_type == "rectangle":
        text, score = paddle_ocr_line(flattened_plate)
        # Validate character count for rectangle plates
        if len(text) in [7, 8]:
            text = fix_license_plate(text, number_to_char, char_to_number, third_char_fixes, invalid_3rd_chars)
            if validate_license_plate(text, plate_type):
                plate_text, ocr_confidence = text, score
    
    elif plate_type == "square":
        # Split the flattened plate for square plates
        line1, line2 = split_license_plate(flattened_plate)
        
        if line1.size > 0 and line2.size > 0:
            text1, score1 = paddle_ocr_line(line1)
            text2, score2 = paddle_ocr_line(line2)
            
            # Lưu giữ text gốc của 2 dòng
            original_text1 = text1
            original_text2 = text2
            
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
                    plate_text, ocr_confidence = text, score
    
    found_plates = []
    
    if plate_text:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Thêm original_text1 và original_text2 vào found_plates cho biển số vuông
        if plate_type == "square":
            found_plates.append((plate_text, plate_type, ocr_confidence, timestamp, original_text1, original_text2))
        else:
            found_plates.append((plate_text, plate_type, ocr_confidence, timestamp))
        
        # Display the plate text on the image
        # Hiển thị biển số vuông thành 2 dòng
        if plate_type == "square":
            cv2.putText(process_frame, f"Plate: {original_text1}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(process_frame, f"      {original_text2}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(process_frame, f"Type: {plate_type}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(process_frame, f"Conf: {ocr_confidence}%", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(process_frame, f"Plate: {plate_text}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(process_frame, f"Type: {plate_type}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(process_frame, f"Conf: {ocr_confidence}%", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write to CSV only if output_csv is provided
        if output_csv is not None:
            with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([plate_text, plate_type, ocr_confidence, timestamp])
        
        # Hiển thị log khác nhau cho biển số vuông và chữ nhật
        if plate_type == "square":
            print(f"Found plate: {original_text1}-{original_text2} ({plate_type}) with confidence {ocr_confidence}")
        else:
            print(f"Found plate: {plate_text} ({plate_type}) with confidence {ocr_confidence}")
    
    # Display the flattened plate
    resized_plate = cv2.resize(flattened_plate, (300, 150))
    h, w, _ = resized_plate.shape
    process_frame[10:10+h, process_frame.shape[1]-w-10:process_frame.shape[1]-10] = resized_plate
    
    # Display the results
    if found_plates:
        if plate_type == "square":
            result_text = f"Found license plate: {original_text1}-{original_text2}"
        else:
            result_text = f"Found license plate: {found_plates[0][0]}"
    else:
        result_text = "No valid license plate detected"
    
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