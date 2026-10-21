# Mốc 3 — Thử nghiệm có kiểm soát

## Chạy và đọc kết quả

Chạy tại thư mục dự án bằng Python trong `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[experiments]"
.\.venv\Scripts\python.exe -m subscription_prediction.experiments --config configs/milestone3.json
```

Đọc `reports/milestone3/metrics.json`, `validation_comparison.csv`,
`calibration_metrics.csv`, `economic_scenarios.csv` và bốn biểu đồ PNG.
Quy trình lưu mô hình/ngưỡng và dấu SHA-256 trong `selection.json` trước khi chấm test.
Sau khi chốt mọi quyết định, lệnh sau chỉ nạp mô hình đã lưu, không huấn luyện lại:

```powershell
.\.venv\Scripts\python.exe -m subscription_prediction.experiments --config configs/milestone3.json --evaluate-test
```

Lệnh test kiểm tra dữ liệu, config, mô hình và báo cáo validation có còn khớp với
SHA-256 đã lưu hay không. Chạy lại lệnh với cùng selection trả lại báo cáo test cũ.
Kết quả test ở `test_metrics.json` và `test_predictions.csv`, tách khỏi báo cáo validation.
Không sửa lựa chọn dựa trên test. Test đã được xem ở baseline/EDA; cần giai đoạn dữ liệu
mới để xác nhận độc lập. Nếu chạy lại validation, báo cáo test cũ chỉ còn hợp lệ khi
`selection_sha256` trong nó khớp SHA-256 của `selection.json` hiện tại.

Thử dự đoán với mô hình mốc 3:

```powershell
.\.venv\Scripts\python.exe -m subscription_prediction.predict --model artifacts/milestone3/model.joblib --input data/examples/customers.csv --output reports/milestone3/customer_priorities.csv
```

## Dòng dữ liệu và lựa chọn mô hình

1. Giữ chia ngoài 70% train, 15% validation, 15% test theo thứ tự dòng.
2. Tạo ba fold mở rộng trong train bằng `TimeSeriesSplit`; kiểm tra cả hai nhãn
   trong phần train và validation của từng fold. `cv_gap` là số dòng bỏ giữa hai phần.
3. Dùng `GridSearchCV` với AP trung bình để chọn siêu tham số cho từng mô hình/nhóm biến.
   Imputer, scaler và encoder được fit lại bên trong mỗi fold.
4. Refit mô hình với tham số tốt nhất trên toàn bộ train; so sánh AP trên validation ngoài.
5. Chọn ứng viên đã tuning có AP validation cao nhất. Baseline được fit lại trên train
   với tham số cũ và cùng tập validation để làm tham chiếu, không tham gia lựa chọn này.
6. Với ứng viên đã chọn, thử sigmoid calibration qua các fold train; chọn raw hoặc
   calibrated bằng Brier score validation thấp hơn.
7. Chọn ngưỡng theo utility mô phỏng trên validation và lưu mọi quyết định trước test.

AP thay đổi theo tỷ lệ yes nền. Đọc `cv_folds.csv` cùng AP từng fold trong
`cv_results.csv`; không coi AP cao hơn ở giai đoạn khác là bằng chứng tự động về cải thiện.
Các fold theo số dòng không phải các khoảng thời gian dài bằng nhau. CSV không có
ngày đầy đủ để chia theo tháng/năm hoặc áp dụng gap theo ngày. `cv_gap=0` mặc định
chưa giải quyết độ trễ có nhãn, các cuộc gọi sát nhau hoặc khách hàng lặp giữa các tập.

## Các nhóm biến và mất cân bằng nhãn

| Nhóm | Cột bỏ thêm ngoài duration | Giả định |
| --- | --- | --- |
| scheduled_assumptions | Không | campaign là số thứ tự cuộc gọi sắp thực hiện; lịch/kênh đã biết; chỉ báo kinh tế đã công bố |
| without_campaign | campaign | Không giả định campaign có sẵn trước cuộc gọi |
| customer_history_only | campaign, contact, month, day_of_week và năm chỉ báo kinh tế | Dùng hồ sơ khách hàng và lịch sử chiến dịch trước |

Tất cả nhóm đều yêu cầu hồ sơ/lịch sử được đóng băng trước lúc chấm điểm. Nhóm
`customer_history_only` giảm số giả định, chưa chứng minh dữ liệu hoàn toàn không lọt thông tin.
Không có dữ liệu vận hành để xác minh thời điểm công bố chỉ báo kinh tế hoặc thời điểm
cập nhật hồ sơ. Ablation đo độ nhạy của mô hình, không xác minh tính sẵn có của biến.
`pdays=999` và literal `unknown` giữ cách xử lý baseline trong thí nghiệm này.

Grid thử `class_weight=None` và `"balanced"` cho Logistic Regression và Random Forest.
Đọc các dòng cùng C/max_depth trong `cv_results.csv` để so sánh trọng số lớp.
Weighting có thể thay đổi score và calibration; không mặc định balanced sẽ tốt hơn.

## Calibration và ngưỡng

`CalibratedClassifierCV(method="sigmoid", ensemble=True, cv=folds)` học từng mô hình
trên phần đầu fold, rồi calibrator trên phần tiếp theo, chỉ trong train. Khi dự đoán,
nó lấy trung bình các mô hình đã calibration. Khác biệt so với raw có cả tác động ensemble;
không diễn giải toàn bộ thay đổi là tác động calibration đơn thuần.

Đọc Brier score và log loss (thấp hơn tốt hơn), đường reliability và số mẫu từng bin.
Brier/log loss còn phản ánh khả năng phân biệt và tỷ lệ nhãn; Brier thấp hơn không đủ
chứng minh calibration tốt. Nhóm ít mẫu dao động mạnh. Validation được dùng lại cho
chọn mô hình, dạng score và ngưỡng, nên kết quả là phân tích khám phá.

Ngưỡng kinh tế tối đa hóa:

```text
utility_proxy = số yes trong nhóm liên hệ × giá trị ròng mỗi đăng ký
                − số người liên hệ × chi phí mỗi cuộc gọi
```

Chi phí và giá trị phải cùng đơn vị. Giá trị ròng mỗi đăng ký ở đây chưa trừ chi phí
cuộc gọi, vì công thức trừ riêng chi phí này. Ba kịch bản minh họa dùng tỷ lệ chi phí:giá trị
1:5, 1:10 và 1:20; kịch bản chính 1:10 được ghi trong config trước khi chạy.
Đây là mô phỏng dựa trên nhãn lịch sử, không phải tác động nhân quả hoặc doanh thu tăng thêm.
Không áp dụng công thức ngưỡng xác suất cost/value trực tiếp cho score chưa được xác nhận.

Thuật toán xét các ngưỡng tại score khác nhau, giới hạn số cuộc gọi validation ở
`ceil(N × contact_fraction)`, không tách nhóm bằng score, và luôn xét phương án không gọi.
Nếu lợi ích bằng nhau, chọn ít cuộc gọi hơn. `threshold_comparison.csv` so sánh với
ngưỡng tối đa F1; ngưỡng F1 có thể vượt ngân sách.
Ngưỡng cố định không bảo đảm giới hạn 20% ở tập mới: muốn giữ ngân sách vận hành,
cần thêm giới hạn top-k theo từng đợt khách hàng và đánh giá chính sách đó riêng.

## Phân tích lỗi và tầm quan trọng biến

- `group_errors.csv`: precision, recall, FP, FN, tỷ lệ nhãn và số mẫu theo nghề, học vấn,
  nhóm tuổi, kênh, tháng, kết quả chiến dịch trước và có/không lịch sử liên hệ.
  Nhóm dưới 100 mẫu được đánh dấu; metric không xác định được để trống.
- `temporal_validation.csv`: các khối dòng liên tiếp của validation, kèm AP, tỷ lệ yes,
  precision/recall và lift theo tỷ lệ yes của từng khối. Khối dòng không phải tháng/năm.
- `validation_predictions.csv`: nhãn thật, score và đánh dấu FP/FN theo source_row.
  source_row không phải customer ID.
- `permutation_importance.csv`: mức giảm AP khi hoán vị từng cột gốc trên validation,
  trung bình và độ lệch chuẩn qua năm lần. Đây là mức phụ thuộc của mô hình, không phải
  quan hệ nhân quả; biến tương quan có thể chia sẻ thông tin và che tầm quan trọng nhau.

## Mã nguồn và kiểm tra

- `tuning.py`: `temporal_folds`, `tune_pipeline`.
- `evaluation.py`: thêm `calibration_metrics`, `economic_metrics`, `select_economic_threshold`.
- `diagnostics.py`: reliability bins, phân tích nhóm/thời gian và xuất biểu đồ.
- `experiments.py`: điều phối thí nghiệm, đóng băng lựa chọn và chấm test riêng.
- `train.py`: tiếp tục là lệnh baseline mốc 2.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm tra bao gồm fold theo thời gian/gap, nhóm chỉ có một nhãn, score bằng nhau,
phương án không gọi và đối chiếu thuật toán ngưỡng với vét cạn. Kiểm tra tích hợp
dùng nhãn test không thể truy cập trong bước validation, xác minh tuning chỉ nhận train,
chấm test không fit lại và chặn dữ liệu đã đổi sau khi đóng băng lựa chọn.

## Nguồn phương pháp

- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing);
  đối chiếu mô tả bank-additional trong gói dữ liệu, vì trang có nhiều phiên bản dataset.
