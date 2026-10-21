# Notebook khám phá dữ liệu

Mở `01_eda.ipynb` để đọc EDA đã chạy trên dữ liệu thật. Notebook bao gồm:

- Kích thước, kiểu dữ liệu, các giá trị `unknown`, tỷ lệ nhãn `yes/no`.
- Phân bố tuổi, nghề nghiệp, lịch sử chiến dịch; so sánh các giai đoạn.
- Kiểm tra bản ghi trùng và ý nghĩa `pdays=999`.
- Danh sách biến khả dụng tại thời điểm chấm điểm trước cuộc gọi.

Không điều chỉnh mô hình dựa trên kết quả tập test. Đặt logic xử lý tái sử dụng trong `src/`.

Phân tích quan hệ biến–nhãn chỉ thực hiện trên train. Validation/test chỉ được kiểm tra
chất lượng và thay đổi phân bố tổng. Baseline đã từng đánh giá test trước EDA;
notebook ghi nhận giới hạn này, không chọn mô hình hoặc ngưỡng từ test.

## Chạy lại

Chạy từ thư mục dự án sau khi đã tải `data/raw/bank-additional-full.csv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[eda]"
.\.venv\Scripts\python.exe tools/run_eda.py
```

Script dùng đúng Python đang gọi để chạy kernel, không cần cài kernel toàn cục.
Nếu chạy trong VS Code/Jupyter, chọn interpreter `.venv` và chạy tất cả cell.

- `docs/eda_findings.md`: nhận xét tiếng Việt từ số liệu của lần chạy.
- `reports/eda/`: bảng CSV, sáu biểu đồ PNG và manifest chứa hash CSV/phiên bản.
- `src/subscription_prediction/eda.py`: các hàm lập bảng dùng lại được.

`reports/eda/` được Git ignore theo quy tắc hiện tại; notebook lưu sẵn output để đọc.
Chạy lại sẽ cập nhật notebook, báo cáo và các file EDA, giữ nguyên CSV gốc/mô hình.
