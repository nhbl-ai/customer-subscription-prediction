# Nhận xét EDA — Bank Marketing

Sinh từ notebooks/01_eda.ipynb và configs/default.json.
Khám phá quan hệ biến–nhãn trên train; validation/test chỉ kiểm tra chất lượng và phân bố tổng.
Baseline đã chấm test trước EDA, nên test không còn hoàn toàn chưa được xem.
Không chọn mô hình, biến, ngưỡng hoặc siêu tham số từ test trong báo cáo này.

## Dữ liệu và chất lượng

- File bank-additional-full.csv: 41,188 dòng, 21 cột gồm nhãn.
- Sau khi loại duration và nhãn y: 19 biến đầu vào.
- NaN toàn bộ file: 0 ô.
- Dòng trùng đầy đủ sau lần đầu: 12 toàn bộ file; 9 train.
- Thiếu customer ID, không xác nhận được cùng người giữa các tập. Không tự xóa dòng trùng hoặc coi source_row là customer ID.

Các giá trị unknown trên train (khác NaN):

- **default**: 7,302 dòng (25.33%).
- **education**: 1,183 dòng (4.10%).
- **housing**: 694 dòng (2.41%).
- **loan**: 694 dòng (2.41%).
- **job**: 249 dòng (0.86%).
- **marital**: 49 dòng (0.17%).

Baseline hiện giữ unknown như nhóm phân loại. Các cách xử lý khác cần thử có kiểm soát.

## Mất cân bằng nhãn và thời gian

| Tập | Số dòng | Số yes | Tỷ lệ yes |
| --- | ---: | ---: | ---: |
| train | 28,831 | 1,606 | 5.57% |
| validation | 6,178 | 658 | 10.65% |
| test | 6,179 | 2,376 | 38.45% |

Luôn đoán no đạt accuracy train 94.43%, nhưng bỏ sót toàn bộ yes.
Bên trong train, 10 khối dòng liên tiếp có tỷ lệ yes từ 2.43% đến 12.73%.
Các khối là vị trí dòng, không phải tháng/năm đầy đủ.

Các khác biệt phân loại lớn nhất đo được:

- month (validation so với train): total variation = 0.701; nhóm chưa có trong train = 0.00%.
- contact (validation so với train): total variation = 0.402; nhóm chưa có trong train = 0.00%.
- poutcome (test so với train): total variation = 0.396; nhóm chưa có trong train = 0.00%.
- contact (test so với train): total variation = 0.353; nhóm chưa có trong train = 0.00%.

Tỷ lệ nhãn và phân bố đầu vào khác nhau giữa các giai đoạn. Đây là bằng chứng mô tả dịch chuyển phân bố,
chưa đủ phân biệt label shift, concept drift hoặc thay đổi cách chọn khách hàng.
Không kết luận AP test cao hơn validation là cải thiện: AP phụ thuộc tỷ lệ yes nền.
Đánh giá theo thời gian, average precision và lift so với baseline ngẫu nhiên cùng tập.

## Tuổi, nghề và lịch sử chiến dịch — train

- Tuổi: median 39, khoảng 18–95.
- Nghề nhiều nhất: admin., 7,107 dòng.
- campaign: median 2, tối đa 56; có đuôi dài.
- pdays=999: 28,778/28,831 dòng (99.82%).
- Chỉ 53 dòng có pdays khác 999; không diễn giải 999 là số ngày hay dùng trung bình nguyên trạng như số ngày điển hình.
- previous=0: 27,829 dòng (96.52%).

Quan hệ nhãn theo nghề, học vấn và kết quả chiến dịch trước chỉ khảo sát trên train.
Đọc cùng số mẫu; nhóm nhỏ có thể dao động mạnh. Không kết luận nhân quả từ các quan hệ này.

## Biến có sẵn trước cuộc gọi

Giả định đã lập lịch, chưa gọi. duration chỉ biết sau cuộc gọi và đã bị loại.
campaign trong UCI tính cả lần liên hệ cuối: cần xác minh nó là số thứ tự cuộc gọi sắp thực hiện,
không phải tổng số liên hệ phát sinh về sau.
contact/month/day_of_week chỉ dùng khi khớp lịch/kênh dự kiến đã biết trước chấm điểm.
pdays/previous/poutcome cần lịch sử đóng băng trước cuộc gọi.
Chỉ báo kinh tế cần phiên bản đã công bố lúc dự đoán, có xem độ trễ và sửa số liệu.
File lịch sử không đủ chứng minh các điều kiện này; xem feature_availability.csv.

## Hướng thử nghiệm ghi nhận khi hoàn thành EDA

Tuning, ablation nhóm biến, calibration và phân tích lỗi đã được thực hiện tiếp ở
[Mốc 3](milestone3_findings.md). Danh sách dưới đây giữ bối cảnh của lần EDA;
xử lý sentinel riêng và xác nhận trên dữ liệu mới vẫn là hướng mở rộng.

1. Ghi rõ giả định campaign và các biến lịch/kênh; thiết kế thử nghiệm có/không có nhóm biến cần xác minh.
2. Thêm TimeSeriesSplit trên train để chọn siêu tham số; kiểm tra mỗi fold có hai nhãn và giữ thứ tự thời gian.
3. Chọn mô hình/ngưỡng trên validation; bổ sung precision–recall và permutation importance trên train/validation.
4. Thử xử lý riêng pdays sentinel hoặc unknown nếu có cơ sở; không thay baseline chỉ vì nhìn test.
5. Giữ test cho đánh giá cuối. Vì test đã được xem, cần một giai đoạn dữ liệu mới nếu muốn xác nhận độc lập về sau.

Dữ liệu chiến dịch lịch sử tại Bồ Đào Nha không đại diện mọi thị trường; notebook không chứng minh hiệu quả triển khai.

## Chạy lại và nguồn

Chạy từ thư mục dự án:

    .\.venv\Scripts\python.exe -m pip install -e ".[eda]"
    .\.venv\Scripts\python.exe tools/run_eda.py

Bảng và sáu biểu đồ: reports/eda/ (Git ignore, có thể sinh lại).
Notebook lưu sẵn kết quả cell để đọc trực tiếp.
SHA-256: 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8.
Phiên bản: {"pandas": "2.3.3", "numpy": "2.5.3", "scikit-learn": "1.9.1", "matplotlib": "3.11.2", "nbclient": "0.11.0"}; Python 3.13.9.

Nguồn: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
Moro, Rita & Cortez (2014), [DOI](https://doi.org/10.24432/C5K306), CC BY 4.0.
Ý nghĩa biến đối chiếu bank-additional-names.txt trong
[gói UCI chính thức](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).
UCI mô tả bản bank-additional-full.csv theo thứ tự thời gian từ 05/2008 đến 11/2010.
