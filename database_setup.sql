CREATE DATABASE IF NOT EXISTS parking_management;
USE parking_management;

CREATE TABLE IF NOT EXISTS xe_dang_gui (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bien_so_xe VARCHAR(20) NOT NULL,
    thoi_gian_vao DATETIME NOT NULL,
    thoi_gian_ra DATETIME NULL,
    INDEX idx_bien_so (bien_so_xe),
    INDEX idx_thoi_gian (thoi_gian_vao, thoi_gian_ra)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;