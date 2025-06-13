import mysql.connector
from mysql.connector import Error

# Cấu hình kết nối MySQL - Điều chỉnh thông tin này theo cài đặt MySQL của bạn
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'vinh2004'
}

def setup_database():
    """Tạo database và các bảng cần thiết nếu chưa tồn tại"""
    connection = None
    cursor = None
    
    try:
        # Kết nối đến MySQL
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Thiết lập múi giờ GMT+7 mặc định
        cursor.execute("SET GLOBAL time_zone = '+7:00'")
        cursor.execute("SET time_zone = '+7:00'")
        print("MySQL time zone set to GMT+7")
        
        # Tạo database
        cursor.execute("CREATE DATABASE IF NOT EXISTS parking_management")
        print("Database 'parking_management' created or already exists")
        
        # Sử dụng database
        cursor.execute("USE parking_management")
        
        # Tạo bảng xe_dang_gui
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS xe_dang_gui (
                id INT AUTO_INCREMENT PRIMARY KEY,
                bien_so_xe VARCHAR(20) NOT NULL,
                thoi_gian_vao DATETIME NOT NULL,
                thoi_gian_ra DATETIME NULL,
                INDEX idx_bien_so (bien_so_xe),
                INDEX idx_thoi_gian (thoi_gian_vao, thoi_gian_ra)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("Table 'xe_dang_gui' created or already exists")

        # Thêm cột ve_thang nếu chưa có - FIX LỖI CÚ PHÁP
        try:
            # Kiểm tra xem cột ve_thang đã tồn tại chưa
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = 'parking_management'
                AND TABLE_NAME = 'xe_dang_gui' 
                AND COLUMN_NAME = 've_thang'
            """)
            column_exists = cursor.fetchone()[0]
            
            # Nếu cột chưa tồn tại, thêm vào
            if column_exists == 0:
                cursor.execute("ALTER TABLE xe_dang_gui ADD COLUMN ve_thang BOOLEAN DEFAULT 0")
                print("Đã thêm cột ve_thang vào bảng xe_dang_gui")
            else:
                print("Cột ve_thang đã tồn tại trong bảng xe_dang_gui")
        except Error as e:
            print(f"Lỗi khi thêm cột ve_thang: {e}")
        
        # Tạo bảng ve_thang
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ve_thang (
                id INT AUTO_INCREMENT PRIMARY KEY,
                bien_so_xe VARCHAR(20) NOT NULL,
                ma_sinh_vien VARCHAR(20) NOT NULL,
                thoi_gian_dang_ki DATETIME NOT NULL,
                thoi_gian_het_han DATETIME NOT NULL,
                INDEX idx_bien_so (bien_so_xe),
                INDEX idx_ma_sinh_vien (ma_sinh_vien),
                INDEX idx_thoi_gian (thoi_gian_dang_ki, thoi_gian_het_han)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("Table 've_thang' created or already exists")

        # Tạo bảng users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                role ENUM('admin', 'staff') NOT NULL DEFAULT 'staff',
                last_login DATETIME NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_username (username),
                INDEX idx_role (role)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("Table 'users' created or already exists")

        # Tạo tài khoản admin mặc định nếu chưa có
        cursor.execute("""
            SELECT COUNT(*) FROM users WHERE role = 'admin'
        """)
        admin_exists = cursor.fetchone()[0]

        if admin_exists == 0:
            import hashlib
            # Mật khẩu mặc định: admin123
            default_password = hashlib.sha256("admin123".encode()).hexdigest()
            
            cursor.execute("""
                INSERT INTO users (username, password, role) 
                VALUES (%s, %s, %s)
            """, ('admin', default_password, 'admin'))
            
            print("Created default admin user (username: admin, password: admin123)")

        # Thêm vào hàm setup_database
        # Tạo bảng user_permissions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_permissions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL UNIQUE,
                entry_access BOOLEAN DEFAULT 1,
                exit_access BOOLEAN DEFAULT 1,
                monthly_ticket_access BOOLEAN DEFAULT 1,
                search_access BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("Table 'user_permissions' created or already exists")
        
        connection.commit()
        print("Database setup completed successfully")
        
    except Error as e:
        print(f"Error setting up database: {e}")
        
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

if __name__ == "__main__":
    setup_database()