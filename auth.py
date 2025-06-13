from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps
from flask import redirect, url_for, flash, session
import hashlib
import mysql.connector
from mysql.connector import Error
from db_utils import get_connection

login_manager = LoginManager()

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role
        self.permissions = None

        # Tự động nạp thông tin phân quyền nếu là staff
        if role == 'staff':
            self.permissions = get_user_permissions(id)
    
    def has_permission(self, permission_type):
        """Kiểm tra user có quyền thực hiện hành động không"""
        # Admin luôn có tất cả quyền
        if self.role == 'admin':
            return True
            
        # Staff cần kiểm tra quyền cụ thể
        if self.role == 'staff' and self.permissions:
            return self.permissions.get(permission_type, False)
            
        return False
    
    @staticmethod
    def get(user_id):
        """Lấy thông tin user theo ID"""
        conn = get_connection()
        if not conn:
            return None
        
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()
            
            if user:
                return User(user['id'], user['username'], user['role'])
            return None
            
        except Error as e:
            print(f"Database error: {e}")
            return None
            
        finally:
            if cursor:
                cursor.close()
            if conn.is_connected():
                conn.close()
    
    @staticmethod
    def authenticate(username, password):
        """Xác thực người dùng"""
        conn = get_connection()
        if not conn:
            return None
        
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Mã hóa mật khẩu để so sánh
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            cursor.execute("""
                SELECT * FROM users 
                WHERE username = %s AND password = %s
            """, (username, hashed_password))
            
            user = cursor.fetchone()
            
            if user:
                # Cập nhật thời gian đăng nhập
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (user['id'],))
                conn.commit()
                
                return User(user['id'], user['username'], user['role'])
            return None
            
        except Error as e:
            print(f"Database error: {e}")
            return None
            
        finally:
            if cursor:
                cursor.close()
            if conn.is_connected():
                conn.close()

# Các decorator cho role
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Quản lý người dùng
def get_all_users():
    """Lấy danh sách tất cả người dùng"""
    conn = get_connection()
    if not conn:
        return []
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users ORDER BY role ASC, username ASC")
        users = cursor.fetchall()
        return users
        
    except Error as e:
        print(f"Database error: {e}")
        return []
        
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def create_user(username, password, role):
    """Tạo người dùng mới"""
    conn = get_connection()
    if not conn:
        return False, "Lỗi kết nối database"
    
    cursor = None
    try:
        cursor = conn.cursor()
        
        # Kiểm tra username đã tồn tại chưa
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
        exists = cursor.fetchone()[0]
        
        if exists > 0:
            return False, f"Tên đăng nhập '{username}' đã tồn tại"
        
        # Mã hóa mật khẩu
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Thêm người dùng mới
        cursor.execute("""
            INSERT INTO users (username, password, role) 
            VALUES (%s, %s, %s)
        """, (username, hashed_password, role))
        
        conn.commit()
        return True, f"Đã tạo tài khoản '{username}' với vai trò '{role}'"
        
    except Error as e:
        print(f"Database error: {e}")
        return False, f"Lỗi database: {str(e)}"
        
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def update_user(user_id, username, password=None):
    """Cập nhật thông tin người dùng"""
    conn = get_connection()
    if not conn:
        return False, "Lỗi kết nối database"
    
    cursor = None
    try:
        cursor = conn.cursor()
        
        # Kiểm tra xem username mới có trùng với username khác không
        cursor.execute("""
            SELECT COUNT(*) FROM users 
            WHERE username = %s AND id != %s
        """, (username, user_id))
        
        exists = cursor.fetchone()[0]
        if exists > 0:
            return False, f"Tên đăng nhập '{username}' đã tồn tại"
        
        # Cập nhật thông tin
        if password:
            # Nếu có cập nhật mật khẩu
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute("""
                UPDATE users SET username = %s, password = %s
                WHERE id = %s
            """, (username, hashed_password, user_id))
        else:
            # Chỉ cập nhật username
            cursor.execute("""
                UPDATE users SET username = %s
                WHERE id = %s
            """, (username, user_id))
        
        conn.commit()
        return True, f"Đã cập nhật thông tin cho tài khoản '{username}'"
        
    except Error as e:
        print(f"Database error: {e}")
        return False, f"Lỗi database: {str(e)}"
        
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def delete_user(user_id):
    """Xóa người dùng"""
    conn = get_connection()
    if not conn:
        return False, "Lỗi kết nối database"
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Kiểm tra xem đây có phải là admin duy nhất không
        cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return False, "Không tìm thấy tài khoản"
        
        if user['role'] == 'admin':
            # Đếm số lượng admin còn lại
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = cursor.fetchone()[0]
            
            if admin_count <= 1:
                return False, "Không thể xóa tài khoản admin duy nhất"
        
        # Lưu tên tài khoản để thông báo
        cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        username = cursor.fetchone()['username']
        
        # Xóa tài khoản
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        
        return True, f"Đã xóa tài khoản '{username}'"
        
    except Error as e:
        print(f"Database error: {e}")
        return False, f"Lỗi database: {str(e)}"
        
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

# ...existing code...

# Thêm các hàm quản lý quyền
def get_user_permissions(user_id):
    """Lấy thông tin phân quyền của user"""
    conn = get_connection()
    if not conn:
        return None
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM user_permissions WHERE user_id = %s
        """, (user_id,))
        
        permissions = cursor.fetchone()
        
        # Nếu chưa có bản ghi phân quyền, tạo mặc định với đầy đủ quyền
        if not permissions:
            cursor.execute("""
                INSERT INTO user_permissions (user_id, entry_access, exit_access, monthly_ticket_access, search_access)
                VALUES (%s, 1, 1, 1, 1)
            """, (user_id,))
            conn.commit()
            
            # Lấy lại sau khi đã thêm
            cursor.execute("""
                SELECT * FROM user_permissions WHERE user_id = %s
            """, (user_id,))
            permissions = cursor.fetchone()
        
        return permissions
        
    except Error as e:
        print(f"Database error: {e}")
        return None
        
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

def update_user_permissions(user_id, entry_access, exit_access, monthly_ticket_access, search_access):
    """Cập nhật phân quyền cho user"""
    conn = get_connection()
    if not conn:
        return False, "Lỗi kết nối database"
    
    cursor = None
    try:
        cursor = conn.cursor()
        
        # Kiểm tra user tồn tại
        cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return False, "Không tìm thấy tài khoản"
            
        username = user[0]
        
        # Kiểm tra đã có bản ghi phân quyền chưa
        cursor.execute("SELECT COUNT(*) FROM user_permissions WHERE user_id = %s", (user_id,))
        exists = cursor.fetchone()[0]
        
        if exists > 0:
            # Cập nhật phân quyền
            cursor.execute("""
                UPDATE user_permissions 
                SET entry_access = %s, exit_access = %s, monthly_ticket_access = %s, search_access = %s
                WHERE user_id = %s
            """, (entry_access, exit_access, monthly_ticket_access, search_access, user_id))
        else:
            # Tạo mới phân quyền
            cursor.execute("""
                INSERT INTO user_permissions (user_id, entry_access, exit_access, monthly_ticket_access, search_access)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, entry_access, exit_access, monthly_ticket_access, search_access))
        
        conn.commit()
        return True, f"Đã cập nhật phân quyền cho tài khoản '{username}'"
        
    except Error as e:
        print(f"Database error: {e}")
        return False, f"Lỗi database: {str(e)}"
        
    finally:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()

# Thêm các decorator để kiểm tra quyền
def entry_permission_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or (current_user.role == 'staff' and not current_user.has_permission('entry_access')):
            # flash('Bạn không có quyền truy cập chức năng này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def exit_permission_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or (current_user.role == 'staff' and not current_user.has_permission('exit_access')):
            flash('Bạn không có quyền truy cập chức năng này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def monthly_ticket_permission_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or (current_user.role == 'staff' and not current_user.has_permission('monthly_ticket_access')):
            flash('Bạn không có quyền truy cập chức năng này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def search_permission_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or (current_user.role == 'staff' and not current_user.has_permission('search_access')):
            flash('Bạn không có quyền truy cập chức năng này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function