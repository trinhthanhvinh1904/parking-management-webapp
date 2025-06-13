from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, session, send_file
import cv2
import os
import time
# from model import process_frame, init_csv
from updated_model import process_frame, init_csv
import numpy as np
import csv
from db_utils import check_and_record_entry, check_and_record_exit, check_and_register_monthly_ticket
from datetime import datetime, timedelta
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from auth import User, admin_required, get_all_users, create_user, update_user, delete_user, entry_permission_required, exit_permission_required, monthly_ticket_permission_required, search_permission_required, update_user_permissions, get_user_permissions
import hashlib
import secrets
import pandas as pd
import io
import base64

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Tạo secret key an toàn

# Đường dẫn camera và CSV
ENTRY_CAMERA = "http://192.168.100.9:8080/video"  # Camera cổng vào
EXIT_CAMERA = "http://192.168.100.9:8080/video"   # Camera cổng ra - thay đổi URL nếu cần
ENTRY_CSV = "entry_log.csv"  # Log cổng vào
EXIT_CSV = "exit_log.csv"    # Log cổng ra

# Global variables
entry_camera = None           # Camera object cổng vào
exit_camera = None            # Camera object cổng ra
entry_camera_active = False   # Trạng thái camera cổng vào
exit_camera_active = False    # Trạng thái camera cổng ra
entry_current_frame = None    # Frame hiện tại cổng vào
exit_current_frame = None     # Frame hiện tại cổng ra

# Cấu hình login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Vui lòng đăng nhập để truy cập trang này.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)
# Camera cổng vào
def get_entry_camera():
    """Khởi tạo và trả về camera object cho cổng vào"""
    global entry_camera, entry_camera_active
    
    if entry_camera is None:
        # Khởi tạo camera nếu chưa có
        entry_camera = cv2.VideoCapture(ENTRY_CAMERA)
        
        if not entry_camera.isOpened():
            print("Error: Could not open entry camera.")
            entry_camera = None
            entry_camera_active = False
            return None
        
        entry_camera_active = True
        print("Entry camera initialized successfully")
    
    return entry_camera

# Camera cổng ra
def get_exit_camera():
    """Khởi tạo và trả về camera object cho cổng ra"""
    global exit_camera, exit_camera_active
    
    if exit_camera is None:
        # Khởi tạo camera nếu chưa có
        exit_camera = cv2.VideoCapture(EXIT_CAMERA)
        
        if not exit_camera.isOpened():
            print("Error: Could not open exit camera.")
            exit_camera = None
            exit_camera_active = False
            return None
        
        exit_camera_active = True
        print("Exit camera initialized successfully")
    
    return exit_camera

# Stream cổng vào
def gen_entry_frames():
    """Generator để stream video từ camera cổng vào - CHỈ để hiển thị, KHÔNG phân tích"""
    global entry_camera_active, entry_current_frame
    
    # Tạo frame trống ban đầu
    blank_frame = create_blank_frame("Đang kết nối camera cổng vào...")
    
    while True:
        # Lấy camera
        cam = get_entry_camera()
        if cam is None or not entry_camera_active:
            # Không có camera, hiển thị frame trống
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
            time.sleep(0.5)  # Đợi lâu hơn khi không có camera
            continue
        
        # Đọc frame từ camera
        ret, frame = cam.read()
        
        if not ret:
            # Nếu không đọc được frame, hiển thị thông báo lỗi
            error_frame = create_blank_frame("Lỗi camera cổng vào - Đang thử lại...")
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)  # Đợi lâu hơn khi có lỗi
            continue
        
        # Lưu frame hiện tại để xử lý khi nhấn nút
        entry_current_frame = frame.copy()
        
        # Gửi frame tới client
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
        
        # Đợi một khoảng thời gian để giảm tải CPU
        time.sleep(0.03)

# Stream cổng ra
def gen_exit_frames():
    """Generator để stream video từ camera cổng ra - CHỈ để hiển thị, KHÔNG phân tích"""
    global exit_camera_active, exit_current_frame
    
    # Tạo frame trống ban đầu
    blank_frame = create_blank_frame("Đang kết nối camera cổng ra...")
    
    while True:
        # Lấy camera
        cam = get_exit_camera()
        if cam is None or not exit_camera_active:
            # Không có camera, hiển thị frame trống
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
            time.sleep(0.5)  # Đợi lâu hơn khi không có camera
            continue
        
        # Đọc frame từ camera
        ret, frame = cam.read()
        
        if not ret:
            # Nếu không đọc được frame, hiển thị thông báo lỗi
            error_frame = create_blank_frame("Lỗi camera cổng ra - Đang thử lại...")
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)  # Đợi lâu hơn khi có lỗi
            continue
        
        # Lưu frame hiện tại để xử lý khi nhấn nút
        exit_current_frame = frame.copy()
        
        # Gửi frame tới client
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
        
        # Đợi một khoảng thời gian để giảm tải CPU
        time.sleep(0.03)

def create_blank_frame(message):
    """Tạo frame trắng với thông điệp"""
    blank_img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(blank_img, message, (150, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', blank_img)
    return buffer.tobytes()

# Routes cho authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Xử lý đăng nhập"""
    # Nếu người dùng đã đăng nhập, chuyển hướng đến trang chủ
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    error = None
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            error = 'Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.'
        else:
            user = User.authenticate(username, password)
            if user:
                login_user(user)
                
                # Chuyển hướng đến trang được yêu cầu trước đó (nếu có)
                next_page = request.args.get('next')
                if not next_page or next_page.startswith('/'):
                    next_page = url_for('index')
                
                return redirect(next_page)
            else:
                error = 'Tên đăng nhập hoặc mật khẩu không đúng.'
    
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    """Xử lý đăng xuất"""
    logout_user()
    return redirect(url_for('login'))

# Routes cho quản lý người dùng (chỉ dành cho admin)
@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Trang quản lý người dùng"""
    users = get_all_users()
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/add', methods=['POST'])
@login_required
@admin_required
def admin_users_add():
    """Thêm người dùng mới"""
    username = request.form.get('username')
    password = request.form.get('password')
    role = request.form.get('role')
    
    if not username or not password:
        flash('Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.', 'danger')
        return redirect(url_for('admin_users'))
    
    success, message = create_user(username, password, role)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/users/edit/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_users_edit(user_id):
    """Chỉnh sửa thông tin người dùng"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username:
        flash('Tên đăng nhập không được để trống.', 'danger')
        return redirect(url_for('admin_users'))
    
    # Nếu password rỗng, không cập nhật password
    success, message = update_user(user_id, username, password if password else None)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_users_delete(user_id):
    """Xóa người dùng"""
    success, message = delete_user(user_id)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(url_for('admin_users'))


# Routes chung
@app.route('/')
@login_required
def index():
    """Trang chủ với lựa chọn cổng vào/ra"""
    return render_template('home.html')

# Routes cổng vào
@app.route('/entry')
@login_required
@entry_permission_required
def entry():
    """Hiển thị trang cổng vào"""
    # Khởi tạo camera cổng vào
    get_entry_camera()
    return render_template('entry.html')

@app.route('/entry_video_feed')
@entry_permission_required
@login_required
def entry_video_feed():
    """Route cho video stream cổng vào"""
    return Response(gen_entry_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/entry_capture_frame', methods=['POST'])
@login_required
@entry_permission_required
def entry_capture_frame():
    """
    Chụp và xử lý một frame từ camera cổng vào khi người dùng nhấn space
    Trả về kết quả nhưng KHÔNG lưu vào CSV ngay
    """
    global entry_current_frame
    
    # Đảm bảo camera đang hoạt động
    if not entry_camera_active or entry_current_frame is None:
        return jsonify({
            "status": "error", 
            "message": "Camera cổng vào không sẵn sàng hoặc không có frame để xử lý"
        })
    
    try:
        # Tạo bản sao của frame hiện tại để xử lý
        frame_to_process = entry_current_frame.copy()
        
        # Xử lý frame
        _, found_plates = process_frame(frame_to_process, None)  # Truyền None để không lưu CSV
        
        # Chuyển đổi frame sang base64 để trả về client
        _, buffer = cv2.imencode('.jpg', frame_to_process)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Trả về kết quả cho frontend
        plates_data = []
        for plate in found_plates:
            plate_info = {
                "text": plate[0],
                "type": plate[1],
                "confidence": plate[2]
            }
            
            # Thêm thông tin text dòng 1 và dòng 2 cho biển số vuông 
            if plate[1] == "square" and len(plate) > 4:
                plate_info["text_line1"] = plate[4]
                plate_info["text_line2"] = plate[5]
            
            plates_data.append(plate_info)
        
        return jsonify({
            "status": "success",
            "message": f"Found {len(found_plates)} license plate(s)",
            "plates": plates_data,
            "frame_base64": frame_base64
        })
        
    except Exception as e:
        # Xử lý lỗi
        return jsonify({"status": "error", "message": f"Lỗi khi xử lý: {str(e)}"})

@app.route('/entry_save_plate', methods=['POST'])
@login_required
@entry_permission_required
def entry_save_plate():
    """
    Lưu thông tin biển số vào CSV và database sau khi đã xác nhận (cổng vào)
    """
    try:
        data = request.json
        plate_text = data.get('text', '')
        plate_type = data.get('type', '')
        confidence = data.get('confidence', 0)
        client_timestamp = data.get('client_timestamp')
        
        # Sử dụng timestamp từ client hoặc tạo mới
        if client_timestamp:
            timestamp = client_timestamp
        else:
            # Tạo timestamp theo múi giờ Việt Nam (UTC+7)
            utc_now = datetime.utcnow()
            vietnam_time = utc_now + timedelta(hours=7)
            timestamp = vietnam_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Lưu vào CSV cổng vào
        with open(ENTRY_CSV, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([plate_text, plate_type, confidence, timestamp])
        
        # Kiểm tra và ghi nhận vào database - truyền timestamp
        success, message, is_duplicate, is_monthly_ticket = check_and_record_entry(plate_text, timestamp)
        
        # Xác định loại vé để hiển thị
        ticket_type = "Vé tháng" if is_monthly_ticket else "Vé ngày"
        
        return jsonify({
            "status": "success" if success else "warning",
            "message": message,
            "timestamp": timestamp,
            "is_duplicate": is_duplicate,
            "is_monthly_ticket": is_monthly_ticket,
            "ticket_type": ticket_type
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi lưu thông tin: {str(e)}"
        })

# Routes cổng ra
@app.route('/exit')
@login_required
@exit_permission_required
def exit():
    """Hiển thị trang cổng ra"""
    # Khởi tạo camera cổng ra
    get_exit_camera()
    return render_template('exit.html')

@app.route('/exit_video_feed')
@login_required
@exit_permission_required
def exit_video_feed():
    """Route cho video stream cổng ra"""
    return Response(gen_exit_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exit_capture_frame', methods=['POST'])
@login_required
@exit_permission_required
def exit_capture_frame():
    """
    Chụp và xử lý một frame từ camera cổng ra khi người dùng nhấn space
    Trả về kết quả nhưng KHÔNG lưu vào CSV ngay
    """
    global exit_current_frame
    
    # Đảm bảo camera đang hoạt động
    if not exit_camera_active or exit_current_frame is None:
        return jsonify({
            "status": "error", 
            "message": "Camera cổng ra không sẵn sàng hoặc không có frame để xử lý"
        })
    
    try:
        # Tạo bản sao của frame hiện tại để xử lý
        frame_to_process = exit_current_frame.copy()
        
        # Xử lý frame
        _, found_plates = process_frame(frame_to_process, None)  # Truyền None để không lưu CSV
        
        # Chuyển đổi frame sang base64 để trả về client
        _, buffer = cv2.imencode('.jpg', frame_to_process)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Trả về kết quả cho frontend
        plates_data = []
        for plate in found_plates:
            plate_info = {
                "text": plate[0],
                "type": plate[1],
                "confidence": plate[2]
            }
            
            # Thêm thông tin text dòng 1 và dòng 2 cho biển số vuông 
            if plate[1] == "square" and len(plate) > 4:
                plate_info["text_line1"] = plate[4]
                plate_info["text_line2"] = plate[5]
            
            plates_data.append(plate_info)
        
        return jsonify({
            "status": "success",
            "message": f"Found {len(found_plates)} license plate(s)",
            "plates": plates_data,
            "frame_base64": frame_base64
        })
        
    except Exception as e:
        # Xử lý lỗi
        return jsonify({"status": "error", "message": f"Lỗi khi xử lý: {str(e)}"})

@app.route('/exit_save_plate', methods=['POST'])
@login_required
@exit_permission_required
def exit_save_plate():
    """
    Lưu thông tin biển số vào CSV và database sau khi đã xác nhận (cổng ra)
    """
    try:
        data = request.json
        plate_text = data.get('text', '')
        plate_type = data.get('type', '')
        confidence = data.get('confidence', 0)
        client_timestamp = data.get('client_timestamp')
        
        # Sử dụng timestamp từ client hoặc tạo mới
        if client_timestamp:
            timestamp = client_timestamp
        else:
            # Tạo timestamp theo múi giờ Việt Nam (UTC+7)
            utc_now = datetime.utcnow()
            vietnam_time = utc_now + timedelta(hours=7)
            timestamp = vietnam_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Lưu vào CSV cổng ra
        with open(EXIT_CSV, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([plate_text, plate_type, confidence, timestamp])
        
        # Kiểm tra và ghi nhận vào database - truyền timestamp
        success, message, entry_time, exit_time, is_monthly_ticket = check_and_record_exit(plate_text, timestamp)

        # Định dạng thời gian vào để khớp với định dạng thời gian ra (YYYY-MM-DD HH:MM:SS)
        formatted_entry_time = entry_time
        if entry_time and isinstance(entry_time, datetime):
            formatted_entry_time = entry_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Xác định loại vé để hiển thị
        ticket_type = "Vé tháng" if is_monthly_ticket else "Vé ngày"
        
        return jsonify({
            "status": "success" if success else "warning",
            "message": message,
            "timestamp": timestamp,
            "entry_time": formatted_entry_time,
            "exit_time": exit_time,
            "is_monthly_ticket": is_monthly_ticket,
            "ticket_type": ticket_type
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi lưu thông tin: {str(e)}"
        })
# Thêm vào trước dòng if __name__ == '__main__':

# Routes cho vé tháng
@app.route('/monthly_ticket')
@login_required
@monthly_ticket_permission_required
def monthly_ticket():
    """Hiển thị trang đăng ký vé tháng"""
    return render_template('monthly_ticket.html')

@app.route('/register_monthly_ticket', methods=['POST'])
@login_required
@monthly_ticket_permission_required
def register_monthly_ticket():
    """Xử lý đăng ký vé tháng"""
    try:
        data = request.json
        bien_so_xe = data.get('bien_so_xe', '')
        ma_sinh_vien = data.get('ma_sinh_vien', '')
        client_timestamp = data.get('client_timestamp')
        
        # Kiểm tra dữ liệu đầu vào
        if not bien_so_xe or not ma_sinh_vien:
            return jsonify({
                "status": "error",
                "message": "Vui lòng nhập đầy đủ biển số xe và mã sinh viên"
            })
        
        # Sử dụng timestamp từ client hoặc tạo mới
        if client_timestamp:
            timestamp = client_timestamp
        else:
            # Tạo timestamp theo múi giờ Việt Nam (UTC+7)
            utc_now = datetime.utcnow()
            vietnam_time = utc_now + timedelta(hours=7)
            timestamp = vietnam_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Kiểm tra và đăng ký vé tháng
        success, message = check_and_register_monthly_ticket(bien_so_xe, ma_sinh_vien, timestamp)
        
        return jsonify({
            "status": "success" if success else "error",
            "message": message,
            "timestamp": timestamp
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi đăng ký vé tháng: {str(e)}"
        })
    
# Routes cho chức năng tra cứu
@app.route('/search_parking')
@login_required
@search_permission_required
def search_parking():
    """Hiển thị trang tra cứu lịch sử ra vào"""
    return render_template('search_parking.html')

@app.route('/search_parking_results', methods=['GET'])
@login_required
@search_permission_required
def search_parking_results():
    """Tìm kiếm và hiển thị kết quả lịch sử ra vào"""
    try:
        # Lấy tham số từ request
        bien_so = request.args.get('bien_so', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        page = int(request.args.get('page', 1))
        
        # Tìm kiếm dữ liệu
        from db_utils import search_parking_history
        results, total_pages = search_parking_history(
            bien_so=bien_so,
            start_date=start_date,
            end_date=end_date,
            page=page
        )
        
        # Format lại thời gian hiển thị (YYYY-MM-DD HH:MM:SS -> DD-MM-YYYY HH:MM:SS)
        formatted_results = []
        for row in results:
            formatted_row = dict(row)
            
            if row['thoi_gian_vao']:
                if isinstance(row['thoi_gian_vao'], str):
                    dt = datetime.strptime(row['thoi_gian_vao'], '%Y-%m-%d %H:%M:%S')
                else:
                    dt = row['thoi_gian_vao']
                formatted_row['thoi_gian_vao_display'] = dt.strftime('%d-%m-%Y %H:%M:%S')
            else:
                formatted_row['thoi_gian_vao_display'] = ''
            
            if row['thoi_gian_ra']:
                if isinstance(row['thoi_gian_ra'], str):
                    dt = datetime.strptime(row['thoi_gian_ra'], '%Y-%m-%d %H:%M:%S')
                else:
                    dt = row['thoi_gian_ra']
                formatted_row['thoi_gian_ra_display'] = dt.strftime('%d-%m-%Y %H:%M:%S')
            else:
                formatted_row['thoi_gian_ra_display'] = 'Chưa ra'
            
            # Xác định loại vé
            formatted_row['loai_ve'] = "Vé tháng" if row['ve_thang'] else "Vé ngày"
            
            formatted_results.append(formatted_row)
        
        # Trả về template với kết quả và thông tin phân trang
        return render_template(
            'search_parking.html',
            results=formatted_results,
            total_pages=total_pages,
            current_page=page,
            bien_so=bien_so,
            start_date=start_date,
            end_date=end_date,
            has_results=True
        )
        
    except Exception as e:
        flash(f"Lỗi khi tìm kiếm: {str(e)}", 'danger')
        return render_template('search_parking.html', error=str(e))

@app.route('/export_parking_excel')
@login_required
@search_permission_required
def export_parking_excel():
    """Xuất kết quả tìm kiếm lịch sử ra vào thành file Excel"""
    try:
        # Lấy tham số từ request
        bien_so = request.args.get('bien_so', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        
        # Tìm kiếm toàn bộ dữ liệu (không phân trang)
        from db_utils import search_parking_history
        results, _ = search_parking_history(
            bien_so=bien_so,
            start_date=start_date,
            end_date=end_date,
            page=1,
            per_page=10000  # Lấy một số lượng lớn kết quả
        )
        
        # Tạo DataFrame từ kết quả
        df_data = []
        for row in results:
            # Format thời gian vào
            if row['thoi_gian_vao']:
                if isinstance(row['thoi_gian_vao'], str):
                    entry_time = datetime.strptime(row['thoi_gian_vao'], '%Y-%m-%d %H:%M:%S')
                else:
                    entry_time = row['thoi_gian_vao']
                entry_time_str = entry_time.strftime('%d-%m-%Y %H:%M:%S')
            else:
                entry_time_str = ''
            
            # Format thời gian ra
            if row['thoi_gian_ra']:
                if isinstance(row['thoi_gian_ra'], str):
                    exit_time = datetime.strptime(row['thoi_gian_ra'], '%Y-%m-%d %H:%M:%S')
                else:
                    exit_time = row['thoi_gian_ra']
                exit_time_str = exit_time.strftime('%d-%m-%Y %H:%M:%S')
            else:
                exit_time_str = 'Chưa ra'
            
            # Xác định loại vé
            ticket_type = "Vé tháng" if row['ve_thang'] else "Vé ngày"
            
            df_data.append({
                'Biển số xe': row['bien_so_xe'],
                'Thời gian vào': entry_time_str,
                'Thời gian ra': exit_time_str,
                'Loại vé': ticket_type
            })
        
        # Tạo DataFrame
        df = pd.DataFrame(df_data)
        
        # Tạo buffer để lưu file Excel
        output = io.BytesIO()
        
        # Tạo ExcelWriter object
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Ghi DataFrame vào Excel
            df.to_excel(writer, sheet_name='Lịch sử ra vào', index=False)
            
            # Điều chỉnh độ rộng cột
            worksheet = writer.sheets['Lịch sử ra vào']
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)
        
        # Đặt con trỏ về đầu file
        output.seek(0)
        
        # Tạo tên file có timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"lich_su_ra_vao_{timestamp}.xlsx"
        
        # Trả về file Excel
        return send_file(
            output, 
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        flash(f"Lỗi khi xuất Excel: {str(e)}", 'danger')
        return redirect(url_for('search_parking'))

@app.route('/search_monthly_tickets')
@login_required
@search_permission_required
def search_monthly_tickets():
    """Hiển thị trang tra cứu lịch sử đăng ký vé tháng"""
    return render_template('search_monthly_tickets.html')

@app.route('/search_monthly_tickets_results', methods=['GET'])
@login_required
@search_permission_required
def search_monthly_tickets_results():
    """Tìm kiếm và hiển thị kết quả lịch sử đăng ký vé tháng"""
    try:
        # Lấy tham số từ request
        bien_so = request.args.get('bien_so', '')
        ma_sinh_vien = request.args.get('ma_sinh_vien', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        page = int(request.args.get('page', 1))
        
        # Tìm kiếm dữ liệu
        from db_utils import search_monthly_tickets
        results, total_pages = search_monthly_tickets(
            bien_so=bien_so,
            ma_sinh_vien=ma_sinh_vien,
            start_date=start_date,
            end_date=end_date,
            page=page
        )
        
        # Format lại thời gian hiển thị (YYYY-MM-DD HH:MM:SS -> DD-MM-YYYY HH:MM:SS)
        formatted_results = []
        for row in results:
            formatted_row = dict(row)
            
            if row['thoi_gian_dang_ki']:
                if isinstance(row['thoi_gian_dang_ki'], str):
                    dt = datetime.strptime(row['thoi_gian_dang_ki'], '%Y-%m-%d %H:%M:%S')
                else:
                    dt = row['thoi_gian_dang_ki']
                formatted_row['thoi_gian_dang_ki_display'] = dt.strftime('%d-%m-%Y %H:%M:%S')
            else:
                formatted_row['thoi_gian_dang_ki_display'] = ''
            
            if row['thoi_gian_het_han']:
                if isinstance(row['thoi_gian_het_han'], str):
                    dt = datetime.strptime(row['thoi_gian_het_han'], '%Y-%m-%d %H:%M:%S')
                else:
                    dt = row['thoi_gian_het_han']
                formatted_row['thoi_gian_het_han_display'] = dt.strftime('%d-%m-%Y %H:%M:%S')
            else:
                formatted_row['thoi_gian_het_han_display'] = ''
            
            formatted_results.append(formatted_row)
        
        # Trả về template với kết quả và thông tin phân trang
        return render_template(
            'search_monthly_tickets.html',
            results=formatted_results,
            total_pages=total_pages,
            current_page=page,
            bien_so=bien_so,
            ma_sinh_vien=ma_sinh_vien,
            start_date=start_date,
            end_date=end_date,
            has_results=True
        )
        
    except Exception as e:
        flash(f"Lỗi khi tìm kiếm: {str(e)}", 'danger')
        return render_template('search_monthly_tickets.html', error=str(e))

@app.route('/export_monthly_tickets_excel')
@login_required
@search_permission_required
def export_monthly_tickets_excel():
    """Xuất kết quả tìm kiếm lịch sử đăng ký vé tháng thành file Excel"""
    try:
        # Lấy tham số từ request
        bien_so = request.args.get('bien_so', '')
        ma_sinh_vien = request.args.get('ma_sinh_vien', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        
        # Tìm kiếm toàn bộ dữ liệu (không phân trang)
        from db_utils import search_monthly_tickets
        results, _ = search_monthly_tickets(
            bien_so=bien_so,
            ma_sinh_vien=ma_sinh_vien,
            start_date=start_date,
            end_date=end_date,
            page=1,
            per_page=10000  # Lấy một số lượng lớn kết quả
        )
        
        # Tạo DataFrame từ kết quả
        df_data = []
        for row in results:
            # Format thời gian đăng kí
            if row['thoi_gian_dang_ki']:
                if isinstance(row['thoi_gian_dang_ki'], str):
                    reg_time = datetime.strptime(row['thoi_gian_dang_ki'], '%Y-%m-%d %H:%M:%S')
                else:
                    reg_time = row['thoi_gian_dang_ki']
                reg_time_str = reg_time.strftime('%d-%m-%Y %H:%M:%S')
            else:
                reg_time_str = ''
            
            # Format thời gian hết hạn
            if row['thoi_gian_het_han']:
                if isinstance(row['thoi_gian_het_han'], str):
                    exp_time = datetime.strptime(row['thoi_gian_het_han'], '%Y-%m-%d %H:%M:%S')
                else:
                    exp_time = row['thoi_gian_het_han']
                exp_time_str = exp_time.strftime('%d-%m-%Y %H:%M:%S')
            else:
                exp_time_str = ''
            
            df_data.append({
                'Biển số xe': row['bien_so_xe'],
                'Mã sinh viên': row['ma_sinh_vien'],
                'Thời gian đăng ký': reg_time_str,
                'Thời gian hết hạn': exp_time_str
            })
        
        # Tạo DataFrame
        df = pd.DataFrame(df_data)
        
        # Tạo buffer để lưu file Excel
        output = io.BytesIO()
        
        # Tạo ExcelWriter object
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Ghi DataFrame vào Excel
            df.to_excel(writer, sheet_name='Lịch sử vé tháng', index=False)
            
            # Điều chỉnh độ rộng cột
            worksheet = writer.sheets['Lịch sử vé tháng']
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)
        
        # Đặt con trỏ về đầu file
        output.seek(0)
        
        # Tạo tên file có timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"lich_su_ve_thang_{timestamp}.xlsx"
        
        # Trả về file Excel
        return send_file(
            output, 
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        flash(f"Lỗi khi xuất Excel: {str(e)}", 'danger')
        return redirect(url_for('search_monthly_tickets'))
# Thêm route cho phân quyền
@app.route('/admin/users/permissions/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_users_permissions(user_id):
    """Cập nhật phân quyền cho người dùng"""
    entry_access = request.form.get('entry_access') == 'on'
    exit_access = request.form.get('exit_access') == 'on'
    monthly_ticket_access = request.form.get('monthly_ticket_access') == 'on'
    search_access = request.form.get('search_access') == 'on'
    
    success, message = update_user_permissions(
        user_id, entry_access, exit_access, monthly_ticket_access, search_access
    )
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'danger')
    
    return redirect(url_for('admin_users'))
@app.route('/admin/get_user_permissions/<int:user_id>')
@login_required
@admin_required
def admin_get_user_permissions(user_id):
    """API lấy thông tin phân quyền người dùng"""
    permissions = get_user_permissions(user_id)
    
    if permissions:
        return jsonify({
            'success': True,
            'permissions': {
                'entry_access': bool(permissions['entry_access']),
                'exit_access': bool(permissions['exit_access']),
                'monthly_ticket_access': bool(permissions['monthly_ticket_access']),
                'search_access': bool(permissions['search_access'])
            }
        })
    
    return jsonify({
        'success': False,
        'message': 'Không tìm thấy thông tin phân quyền'
    })

if __name__ == '__main__':
    # Đảm bảo thư mục output tồn tại
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Khởi tạo các file CSV
    init_csv(ENTRY_CSV)
    init_csv(EXIT_CSV)
    
    # Thông báo cách cấu hình MySQL
    print("LƯU Ý: Hãy cập nhật thông tin kết nối MySQL trong file db_utils.py và setup_database.py")
    print("Sau đó chạy lệnh 'python setup_database.py' để khởi tạo database trước khi chạy app")
    
    # Chạy ứng dụng Flask
    app.run(debug=True, host='0.0.0.0', port=5000)