# Kế hoạch và trạng thái hoàn thành

Cập nhật 21/10/2026. Bốn mốc đã hoàn thành trong phạm vi dự án portfolio học máy.
Đây là trạng thái mã nguồn/tài liệu trong workspace; việc push và đổi visibility GitHub
được thực hiện theo [release guide](release_guide.md).

| Mốc | Trạng thái | Bằng chứng |
| --- | --- | --- |
| 1 — Hiểu dữ liệu | Hoàn thành | [Notebook đã chạy](../notebooks/01_eda.ipynb), [EDA findings](eda_findings.md) |
| 2 — Baseline tái lập | Hoàn thành | [Train CLI](../src/subscription_prediction/train.py), [config](../configs/default.json), baseline tham chiếu trong [bảng validation](results/validation_comparison.csv) |
| 3 — Thử nghiệm có kiểm soát | Hoàn thành | [Hướng dẫn](milestone3_guide.md), [kết quả và giới hạn](milestone3_findings.md), [snapshot](results/README.md) |
| 4 — Portfolio công khai | Hoàn thành | [README](../README.md), [CHANGELOG](../CHANGELOG.md), [portfolio](portfolio_report.md), [model card](model_card.md), [môi trường](../requirements-lock.txt), [CI](../.github/workflows/ci.yml) |

## Mốc 1 — Hiểu dữ liệu

- [x] Notebook EDA có output; kiểm tra schema, nhãn, unknown, trùng dòng và `pdays=999`.
- [x] Xác định thời điểm chấm điểm sau lập lịch, trước cuộc gọi; loại `duration`.
- [x] Ghi giả định về campaign, lịch/kênh và độ trễ chỉ báo kinh tế.
- [x] Phân tích mất cân bằng nhãn và thay đổi phân bố theo thứ tự dòng.

## Mốc 2 — Baseline có thể chạy lại

- [x] Pipeline tiền xử lý chỉ fit trên train; so sánh Dummy/LR/RF bằng AP validation.
- [x] Ngưỡng F1 chọn trên validation, báo cáo test và lift theo ngân sách.
- [x] Lưu artifact và cung cấp CLI chấm điểm/xếp hạng CSV giả lập.
- [x] Kiểm thử luồng train → artifact → inference trong thư mục tạm, không tải dữ liệu thật.

Test đã được xem ở baseline/EDA. Mốc 3 đóng băng lựa chọn trước lần chấm riêng,
nhưng quy trình đó không khôi phục tính độc lập của test đã xem.

## Mốc 3 — Thử nghiệm có kiểm soát

- [x] Ba fold TimeSeriesSplit trong train; 27 ứng viên grid, 81 lần fit fold.
- [x] Ba nhóm biến, hai lựa chọn class_weight; tuning không dùng validation ngoài/test.
- [x] Precision–recall, permutation importance, phân tích lỗi nhóm và khối thời gian.
- [x] So sánh raw/temporal sigmoid; ngưỡng theo ba kịch bản chi phí minh họa.
- [x] Đóng băng mô hình/ngưỡng và hash, chấm test riêng không refit.
- [x] Công bố cả thử nghiệm không cải thiện và các giới hạn của kết quả.

## Mốc 4 — Portfolio

- [x] README tiếng Anh cho người tuyển dụng; báo cáo tiếng Việt có số liệu, hình và ví dụ lỗi.
- [x] Model card, CHANGELOG, license, attribution dữ liệu và hướng dẫn phát hành.
- [x] Snapshot tổng hợp được Git theo dõi; người đọc không cần train để xem kết quả.
- [x] Ghi môi trường bằng pip freeze; thêm CI và kiểm thử không phụ thuộc CSV UCI.
- [x] Demo CLI sử dụng CSV giả lập và model đã train; có hướng dẫn đa nền tảng.
- [x] Câu mô tả CV dùng số liệu validation đã đo, ghi rõ là offline evaluation.

Demo web/API là tùy chọn trong kế hoạch gốc. Bản portfolio này dùng batch CLI demo;
không có ứng dụng đã host hoặc API, không tuyên bố đã triển khai thực tế.

## Hướng mở rộng sau portfolio

- Thu thập giai đoạn mới để xác nhận độc lập; kiểm tra customer ID, nhãn trễ và dữ liệu tại thời điểm chấm.
- Xác minh biến trước cuộc gọi và chi phí/giá trị thực tế cùng bên vận hành.
- Đánh giá chính sách ngân sách theo batch, calibration mới và độ bất định.
- Thử biểu diễn riêng sentinel `pdays=999`, thêm phân tích công bằng theo phạm vi áp dụng.
- Bổ sung giao diện web nếu có nhu cầu; kiểm chứng tác động bằng thiết kế thử nghiệm thực tế.
