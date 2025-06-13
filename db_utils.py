# db_utils.py
import mysql.connector
from mysql.connector import Error
import time
from datetime import datetime, timedelta

# Cấu hình kết nối MySQL - Điều chỉnh thông tin này theo cài đặt MySQL của bạn
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'vinh2004',
    'database': 'parking_management'
}

# def get_connection():
#     """Tạo kết nối đến MySQL database"""
#     try:
#         connection = mysql.connector.connect(**DB_CONFIG)
#         return connection
#     except Error as e:
#         print(f"Error connecting to MySQL database: {e}")
#         return None

def get_connection():
    """Tạo kết nối đến MySQL database với múi giờ GMT+7"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        
        # Thiết lập múi giờ GMT+7 cho phiên kết nối hiện tại
        cursor = connection.cursor()
        cursor.execute("SET time_zone = '+7:00'")
        cursor.close()
        
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def check_and_record_entry(bien_so, timestamp=None):
    """
    Kiểm tra biển số xe khi vào bãi và ghi nhận thời gian vào
    Trả về tuple (success, message, is_duplicate, is_monthly_ticket)
    """
    conn = get_connection()
    if not conn:
        return False, "Lỗi kết nối đến database", False, False
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Kiểm tra biển số đã có và chưa ra khỏi bãi không
        query = """
            SELECT * FROM xe_dang_gui 
            WHERE bien_so_xe = %s 
            ORDER BY thoi_gian_vao DESC 
            LIMIT 1
        """
        cursor.execute(query, (bien_so,))
        result = cursor.fetchone()
        
        # Nếu không được cung cấp timestamp, tạo timestamp hiện tại
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Kiểm tra vé tháng
        is_monthly_ticket = check_monthly_ticket(bien_so, timestamp)
        
        # Nếu không tìm thấy biển số hoặc biển số đã có thời gian ra (đã ra khỏi bãi)
        if not result or result['thoi_gian_ra'] is not None:
            # Tạo bản ghi mới với trạng thái vé tháng
            insert_query = """
                INSERT INTO xe_dang_gui (bien_so_xe, thoi_gian_vao, ve_thang)
                VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (bien_so, timestamp, 1 if is_monthly_ticket else 0))
            conn.commit()
            
            return True, f"Đã ghi nhận xe {bien_so} vào bãi lúc {timestamp}", False, is_monthly_ticket
        else:
            # Biển số đã có và chưa ra khỏi bãi
            return False, f"Xe {bien_so} đã vào bãi từ {result['thoi_gian_vao']} và chưa ra khỏi bãi", True, False
    
    except Error as e:
        print(f"Database error: {e}")
        return False, f"Lỗi database: {str(e)}", False, False
    
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def check_and_record_exit(bien_so, timestamp=None):
    """
    Kiểm tra biển số xe khi ra khỏi bãi và ghi nhận thời gian ra
    Trả về tuple (success, message, entry_time, exit_time, is_monthly_ticket)
    """
    conn = get_connection()
    if not conn:
        return False, "Lỗi kết nối đến database", None, None, False
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Kiểm tra biển số có trong bãi không
        query = """
            SELECT * FROM xe_dang_gui 
            WHERE bien_so_xe = %s AND thoi_gian_ra IS NULL
            ORDER BY thoi_gian_vao DESC 
            LIMIT 1
        """
        cursor.execute(query, (bien_so,))
        result = cursor.fetchone()
        
        # Nếu không được cung cấp timestamp, tạo timestamp hiện tại
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Nếu tìm thấy biển số và chưa có thời gian ra
        if result:
            # Cập nhật thời gian ra
            update_query = """
                UPDATE xe_dang_gui 
                SET thoi_gian_ra = %s
                WHERE id = %s
            """
            cursor.execute(update_query, (timestamp, result['id']))
            conn.commit()
            
            # Lấy thông tin vé tháng từ bản ghi
            is_monthly_ticket = bool(result['ve_thang'])
            
            return True, f"Đã ghi nhận xe {bien_so} ra khỏi bãi", result['thoi_gian_vao'], timestamp, is_monthly_ticket
        else:
            # Không tìm thấy biển số hoặc biển số đã ra khỏi bãi
            return False, f"Xe {bien_so} chưa vào bãi hoặc đã ra khỏi bãi rồi", None, None, False
    
    except Error as e:
        print(f"Database error: {e}")
        return False, f"Lỗi database: {str(e)}", None, None, False
    
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

# Thêm vào cuối file

def check_and_register_monthly_ticket(bien_so_xe, ma_sinh_vien, timestamp=None):
    """
    Kiểm tra và đăng ký vé tháng nếu thỏa mãn điều kiện
    Trả về tuple (success, message)
    """
    conn = get_connection()
    if not conn:
        return False, "Lỗi kết nối đến database"
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Nếu không được cung cấp timestamp, tạo timestamp hiện tại
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Chuyển timestamp thành đối tượng datetime
        if isinstance(timestamp, str):
            dang_ki_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            dang_ki_date = timestamp
        
        # Xác định thời gian hết hạn (12h01 ngày mùng 1 tháng sau)
        if dang_ki_date.month == 12:
            next_month = 1
            next_year = dang_ki_date.year + 1
        else:
            next_month = dang_ki_date.month + 1
            next_year = dang_ki_date.year
        
        # Tạo thời gian hết hạn: 12h01 ngày mùng 1 tháng sau
        het_han_date = datetime(next_year, next_month, 1, 12, 1, 0)
        het_han = het_han_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Ngày đầu tháng hiện tại
        first_day = datetime(dang_ki_date.year, dang_ki_date.month, 1, 0, 0, 0)
        
        # Ngày đầu tháng tiếp theo (để tìm cuối tháng)
        if dang_ki_date.month == 12:
            next_month_start = datetime(dang_ki_date.year + 1, 1, 1, 0, 0, 0)
        else:
            next_month_start = datetime(dang_ki_date.year, dang_ki_date.month + 1, 1, 0, 0, 0)
        
        # Ngày cuối tháng (trừ đi 1 giây từ ngày đầu tháng tiếp theo)
        last_day = next_month_start - timedelta(seconds=1)
        
        # Chuyển thành chuỗi để query
        first_day_str = first_day.strftime("%Y-%m-%d %H:%M:%S")
        last_day_str = last_day.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_str = dang_ki_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Kiểm tra xem sinh viên đã có vé tháng CÒNG HIỆU LỰC trong tháng này chưa
        # Vé còn hiệu lực là vé có thời gian hết hạn SAU thời điểm đăng ký hiện tại
        query = """
            SELECT * FROM ve_thang 
            WHERE ma_sinh_vien = %s 
            AND thoi_gian_het_han > %s
            AND (
                (thoi_gian_dang_ki BETWEEN %s AND %s)  
                OR (thoi_gian_het_han BETWEEN %s AND %s)
                OR (thoi_gian_dang_ki <= %s AND thoi_gian_het_han >= %s)
            )
        """
        cursor.execute(query, (
            ma_sinh_vien, timestamp_str,
            first_day_str, last_day_str, 
            first_day_str, last_day_str,
            first_day_str, last_day_str
        ))
        result = cursor.fetchone()
        
        if result:
            return False, f"Sinh viên {ma_sinh_vien} đã đăng ký vé tháng trong tháng này và vé còn hiệu lực."
        
        # 2. Kiểm tra xem biển số xe đã được người khác đăng ký và còn hiệu lực trong tháng này chưa
        query = """
            SELECT * FROM ve_thang 
            WHERE bien_so_xe = %s 
            AND ma_sinh_vien <> %s
            AND thoi_gian_het_han > %s
            AND (
                (thoi_gian_dang_ki BETWEEN %s AND %s)  
                OR (thoi_gian_het_han BETWEEN %s AND %s)
                OR (thoi_gian_dang_ki <= %s AND thoi_gian_het_han >= %s)
            )
        """
        cursor.execute(query, (
            bien_so_xe, ma_sinh_vien, timestamp_str,
            first_day_str, last_day_str, 
            first_day_str, last_day_str,
            first_day_str, last_day_str
        ))
        result = cursor.fetchone()
        
        if result:
            return False, f"Biển số xe {bien_so_xe} đã được đăng ký bởi sinh viên khác trong tháng này và vé còn hiệu lực."
        
        # 3. Nếu thỏa mãn cả hai điều kiện, tiến hành đăng ký
        insert_query = """
            INSERT INTO ve_thang (bien_so_xe, ma_sinh_vien, thoi_gian_dang_ki, thoi_gian_het_han)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (bien_so_xe, ma_sinh_vien, timestamp_str, het_han))
        conn.commit()
        
        return True, f"Đăng ký vé tháng thành công cho sinh viên {ma_sinh_vien} với biển số xe {bien_so_xe}. Vé có hiệu lực đến {het_han}."
    
    except Error as e:
        print(f"Database error: {e}")
        return False, f"Lỗi database: {str(e)}"
    
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def check_monthly_ticket(bien_so, timestamp):
    """
    Kiểm tra xem biển số có phải là vé tháng hợp lệ còn hạn không
    Trả về True nếu là vé tháng còn hạn, False nếu không phải hoặc hết hạn
    """
    conn = get_connection()
    if not conn:
        return False
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Lấy bản ghi vé tháng mới nhất của biển số xe
        query = """
            SELECT * FROM ve_thang 
            WHERE bien_so_xe = %s 
            ORDER BY thoi_gian_het_han DESC 
            LIMIT 1
        """
        cursor.execute(query, (bien_so,))
        result = cursor.fetchone()
        
        # Nếu không tìm thấy bản ghi vé tháng
        if not result:
            return False
        
        # Chuyển đổi timestamp từ chuỗi sang datetime nếu cần
        if isinstance(timestamp, str):
            timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            timestamp_dt = timestamp
        
        # Kiểm tra thời gian hiện tại có nằm trong khoảng đăng ký và hết hạn không
        if timestamp_dt >= result['thoi_gian_dang_ki'] and timestamp_dt <= result['thoi_gian_het_han']:
            return True
        else:
            return False
    
    except Error as e:
        print(f"Database error when checking monthly ticket: {e}")
        return False
    
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def search_parking_history(bien_so=None, start_date=None, end_date=None, page=1, per_page=10):
    """
    Tìm kiếm lịch sử ra vào bãi đỗ xe
    Parameters:
        bien_so (str): Biển số xe cần tìm
        start_date (str): Ngày bắt đầu (YYYY-MM-DD)
        end_date (str): Ngày kết thúc (YYYY-MM-DD)
        page (int): Số trang
        per_page (int): Số kết quả mỗi trang
    Returns:
        (results, total_pages): Kết quả tìm kiếm và tổng số trang
    """
    conn = get_connection()
    if not conn:
        return [], 0
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Xây dựng câu truy vấn cơ bản
        query = "SELECT * FROM xe_dang_gui WHERE 1=1"
        params = []
        
        # Thêm điều kiện biển số nếu có
        if bien_so:
            query += " AND bien_so_xe LIKE %s"
            params.append(f"%{bien_so}%")
        
        # Thêm điều kiện thời gian nếu có
        if start_date:
            query += " AND thoi_gian_vao >= %s"
            params.append(f"{start_date} 00:00:00")
        
        if end_date:
            query += " AND thoi_gian_vao <= %s"
            params.append(f"{end_date} 23:59:59")
        
        # Đếm tổng số kết quả để tính số trang
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['COUNT(*)']
        
        # Tính tổng số trang
        total_pages = (total_count + per_page - 1) // per_page
        
        # Thêm ORDER BY và LIMIT cho phân trang
        query += " ORDER BY thoi_gian_vao DESC LIMIT %s OFFSET %s"
        offset = (page - 1) * per_page
        params.extend([per_page, offset])
        
        # Thực hiện truy vấn chính
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return results, total_pages
    
    except Error as e:
        print(f"Database error: {e}")
        return [], 0
    
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def search_monthly_tickets(bien_so=None, ma_sinh_vien=None, start_date=None, end_date=None, page=1, per_page=10):
    """
    Tìm kiếm lịch sử đăng ký vé tháng
    Parameters:
        bien_so (str): Biển số xe cần tìm
        ma_sinh_vien (str): Mã sinh viên cần tìm
        start_date (str): Ngày bắt đầu (YYYY-MM-DD)
        end_date (str): Ngày kết thúc (YYYY-MM-DD)
        page (int): Số trang
        per_page (int): Số kết quả mỗi trang
    Returns:
        (results, total_pages): Kết quả tìm kiếm và tổng số trang
    """
    conn = get_connection()
    if not conn:
        return [], 0
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Xây dựng câu truy vấn cơ bản
        query = "SELECT * FROM ve_thang WHERE 1=1"
        params = []
        
        # Thêm điều kiện biển số nếu có
        if bien_so:
            query += " AND bien_so_xe LIKE %s"
            params.append(f"%{bien_so}%")
        
        # Thêm điều kiện mã sinh viên nếu có
        if ma_sinh_vien:
            query += " AND ma_sinh_vien LIKE %s"
            params.append(f"%{ma_sinh_vien}%")
        
        # Thêm điều kiện thời gian nếu có
        if start_date:
            query += " AND thoi_gian_dang_ki >= %s"
            params.append(f"{start_date} 00:00:00")
        
        if end_date:
            query += " AND thoi_gian_dang_ki <= %s"
            params.append(f"{end_date} 23:59:59")
        
        # Đếm tổng số kết quả để tính số trang
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()['COUNT(*)']
        
        # Tính tổng số trang
        total_pages = (total_count + per_page - 1) // per_page
        
        # Thêm ORDER BY và LIMIT cho phân trang
        query += " ORDER BY thoi_gian_dang_ki DESC LIMIT %s OFFSET %s"
        offset = (page - 1) * per_page
        params.extend([per_page, offset])
        
        # Thực hiện truy vấn chính
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return results, total_pages
    
    except Error as e:
        print(f"Database error: {e}")
        return [], 0
    
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()