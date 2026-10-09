# Predicting Customer Service Subscriptions

Khung dự án Machine Learning dự đoán khách hàng có đăng ký dịch vụ hay không,
sử dụng Python và scikit-learn. Bài toán cụ thể ban đầu là đăng ký tiền gửi có kỳ hạn
trong bộ dữ liệu **UCI Bank Marketing**.

**Đầu vào:** thông tin khách hàng và lịch sử chiến dịch, ở thời điểm đã lập lịch nhưng chưa gọi.
**Đầu ra:** điểm dự đoán đăng ký, nhãn yes/no và danh sách ưu tiên liên hệ.

## Cấu trúc

```text
customer-subscription-prediction/
├── configs/default.json         # Đường dẫn, chia dữ liệu, tham số
├── data/
│   ├── raw/                     # CSV gốc từ UCI
│   ├── processed/               # Dành cho các bước mở rộng
│   └── examples/customers.csv   # Dữ liệu giả lập để thử dự đoán
├── src/subscription_prediction/
│   ├── data.py                  # Đọc dữ liệu, kiểm tra nhãn, chia tập
│   ├── modeling.py              # Pipeline và ba mô hình
│   ├── evaluation.py            # Metric, ngưỡng và ngân sách liên hệ
│   ├── train.py                 # Huấn luyện và xuất báo cáo
│   └── predict.py               # Chấm điểm và xếp hạng CSV
├── tools/download_data.py       # Tải CSV công khai từ UCI
├── notebooks/README.md          # Gợi ý notebook EDA
├── docs/project_plan.md         # Các mốc hoàn thiện portfolio
├── artifacts/                   # model.joblib sau khi train
├── reports/                     # Báo cáo và CSV kết quả
├── pyproject.toml
└── requirements.txt
```

## Bắt đầu trên Windows / PowerShell

Yêu cầu Python 3.11 trở lên. Chạy tại thư mục dự án:

```powershell
Set-Location 'D:\Personal projects\customer-subscription-prediction'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/download_data.py
.\.venv\Scripts\python.exe -m subscription_prediction.train --config configs/default.json
.\.venv\Scripts\python.exe -m subscription_prediction.predict --input data/examples/customers.csv --output reports/customer_priorities.csv
```

Dùng trực tiếp Python trong .venv nên không cần thay đổi PowerShell execution policy.
Nếu máy chỉ nhận lệnh `py`, dùng `py -3 -m venv .venv` ở bước tạo môi trường.
Khung không kèm dữ liệu thật hoặc mô hình đã huấn luyện. Tải và cài thư viện cần Internet.

Nếu không tải được bằng script: mở https://archive.ics.uci.edu/dataset/222/bank+marketing,
tải và giải nén **bank-additional-full.csv** vào `data/raw/`.
Không dùng nhầm `bank-full.csv`: hai phiên bản có các cột khác nhau.
Dữ liệu ví dụ là giả lập, không dùng để huấn luyện hoặc báo cáo chất lượng.

## Huấn luyện làm gì?

1. Đọc CSV dấu `;`, ánh xạ y=no thành 0 và y=yes thành 1.
2. Loại `duration`: chỉ biết độ dài cuộc gọi khi cuộc gọi kết thúc.
3. Chia 70% train, 15% validation, 15% test theo thứ tự dòng. UCI mô tả phiên bản
   bank-additional-full.csv được sắp theo thời gian; pipeline không tự suy ra/sắp xếp ngày.
4. Chỉ fit imputer, scaler, encoder và estimator trên train trong Pipeline.
5. So sánh Dummy, Logistic Regression và Random Forest bằng **average precision trên validation**.
6. Chọn ngưỡng tối đa F1 trên validation cho từng mô hình.
7. Chấm mô hình đã chọn trên test và lưu báo cáo. Không refit trên validation trong khung này.

Với CSV khác không có thứ tự thời gian, cân nhắc `split_strategy: stratified` trong config.
Nếu nhiều dòng thuộc cùng khách hàng, cần bổ sung chia theo customer ID để tránh lọt thông tin;
khung hiện tại không suy ra được danh tính khách hàng từ dataset.

## Đọc kết quả

- `artifacts/model.joblib`: pipeline, tên mô hình, ngưỡng, cột đầu vào và phiên bản sklearn.
- `reports/validation_comparison.csv`: so sánh các mô hình trên validation.
- `reports/metrics.json`: cấu hình, kích thước tập, phiên bản và metric validation/test.
- `reports/test_predictions.csv`: nhãn thật và dự đoán cho test.
- `reports/customer_priorities.csv`: xếp hạng khách hàng theo subscription_score giảm dần.

`source_row` là vị trí dòng dữ liệu tính từ 0, không phải customer ID.
Confusion matrix có thứ tự [[TN, FP], [FN, TP]]. Average precision tóm tắt đường precision–recall.
`contact_fraction=0.2` đo precision và lift nếu ưu tiên liên hệ 20% khách hàng có điểm cao nhất.
Baseline ngẫu nhiên có precision kỳ vọng bằng tỷ lệ yes của tập đang đánh giá.
Điểm bằng nhau không có thứ tự ưu tiên có ý nghĩa; đặc biệt chú ý kết quả xếp hạng của Dummy.

## Dự đoán dữ liệu của bạn

CSV đầu vào phải có tất cả cột đã dùng lúc train. Có thể thêm customer_id; cột này
sẽ được giữ trong kết quả, nhưng không dùng cho mô hình nếu không có lúc train.
Không cần cột y hoặc duration. CSV kết quả dùng dấu phẩy.

```powershell
.\.venv\Scripts\python.exe -m subscription_prediction.predict --input data/examples/customers.csv --top-k 2
```

subscription_score là điểm từ predict_proba; các mô hình dùng class_weight chưa được
hiệu chỉnh xác suất nên không diễn giải trực tiếp thành tỷ lệ đăng ký ngoài thực tế.
Chỉ nạp file joblib có nguồn đáng tin vì cơ chế pickle có thể thực thi mã.

## Các bước tiếp theo

Đọc `docs/project_plan.md`: thêm EDA, TimeSeriesSplit, chọn siêu tham số, phân tích lỗi,
giải thích mô hình, calibration và demo. Khung hiện tại cung cấp baseline chạy được;
chưa có tuning, biểu đồ, notebook thực thi hoặc ứng dụng web.

## Nguồn

- Dữ liệu: Moro, S., Rita, P., & Cortez, P. (2014), Bank Marketing,
  UCI Machine Learning Repository, https://doi.org/10.24432/C5K306 — CC BY 4.0.
- scikit-learn: https://scikit-learn.org/stable/getting_started.html

Đây là dự án học tập trên dữ liệu chiến dịch lịch sử tại Bồ Đào Nha. Kết quả không tự
chứng minh hiệu quả trên thị trường hoặc chiến dịch mới; cần đánh giá lại trước khi áp dụng.
