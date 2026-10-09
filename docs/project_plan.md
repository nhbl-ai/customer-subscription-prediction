# Kế hoạch học và hoàn thiện

## Mốc 1 — Hiểu dữ liệu
- Đọc mô tả UCI và tạo notebook EDA.
- Xác định rõ thời điểm dự đoán: sau khi lập lịch, trước khi gọi.
- Loại duration; xem lại campaign và các biến liên hệ theo tình huống sử dụng.
- Ghi nhận dữ liệu lịch sử ở Bồ Đào Nha, không đại diện mọi thị trường.

## Mốc 2 — Baseline có thể chạy lại
- Chạy pipeline mặc định và đọc reports/metrics.json.
- So sánh Dummy, Logistic Regression và Random Forest trên validation.
- Giải thích precision, recall, F1, average precision và lift theo ngân sách.
- Giữ nguyên test đến khi chọn mô hình và ngưỡng xong.

## Mốc 3 — Thử nghiệm có kiểm soát
- Thêm TimeSeriesSplit trên train để chọn siêu tham số.
- Thêm permutation importance và biểu đồ precision–recall.
- Phân tích nhóm bị sai, tác động của mất cân bằng nhãn và thay đổi theo thời gian.
- Chọn ngưỡng theo chi phí cuộc gọi/doanh thu, thay vì chỉ tối đa F1.
- Đánh giá calibration trước khi diễn giải score thành xác suất thực tế.

## Mốc 4 — Portfolio
- Thêm demo Streamlit hoặc API nếu cần.
- Viết báo cáo với số liệu đã đo, dữ liệu, cấu hình, giới hạn và ví dụ lỗi.
- Ghi lại môi trường bằng python -m pip freeze sau khi hoàn thiện.
- Khi viết CV, dùng số liệu thực tế; không tự đặt một mục tiêu accuracy để điền vào CV.
