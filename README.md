# Dash Regression

Ứng dụng web nhỏ viết bằng Python + Dash để minh họa hồi quy tuyến tính.  
Người dùng có thể nhập dữ liệu, chạy mô hình, và xem kết quả trực quan.

---

## 📌 Giới thiệu
Dash Regression là một demo ứng dụng sử dụng [Plotly Dash](https://dash.plotly.com/) để xây dựng giao diện web cho mô hình hồi quy tuyến tính.  
Dự án giúp người học và nhà phát triển dễ dàng hình dung cách tích hợp mô hình học máy vào ứng dụng web.

---

## ⚙️ Yêu cầu hệ thống
- Python >= 3.8
- dash
- plotly
- pandas
- scikit-learn

Cài đặt nhanh:
```bash
pip install -r requirements.txt
```

---
## 🚀 Cài đặt & Chạy
- Clone repo và chạy ứng dụng:

```bat
git clone https://github.com/Thomas131104/dash-regression.git
cd dash-regression
pip install -r requirements.txt
python app.py
```

- Ứng dụng sẽ chạy tại http://127.0.0.1:8050/.

---
## 📊 Cách sử dụng
- Mở trình duyệt tại địa chỉ trên.
- Nhập danh sách x, y
- Chọn chế độ
  - Không nhập: Linear Regression
  - Chọn Lasso: Lasso Regression
  - Chọn Ridge: Ridge Regression
  - Chọn cả Lasso và Ridge: Elastic Net Regression
- Xem kết quả hồi quy và biểu đồ trực quan.

---

## 🖼️ Ví dụ minh họa
Thêm ảnh chụp màn hình giao diện sau khi chạy thử để README hấp dẫn hơn.

--- 

## 📚 Cấu trúc thư mục
Code
- app.py        # File chính chạy Dash app
- assets/       # Chứa CSS hoặc file tĩnh

---

## 🔮 Hướng phát triển
- Hỗ trợ nhiều loại hồi quy khác (đa biến, logistic).
- Cho phép export kết quả ra CSV/Excel.
- Thêm giao diện đẹp hơn với Bootstrap.
