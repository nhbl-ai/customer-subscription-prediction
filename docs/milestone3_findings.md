# Kết quả Mốc 3 — Bank Marketing

Đã chạy `configs/milestone3.json` trên dữ liệu thật của dự án.
Mã nguồn và cách chạy: [milestone3_guide.md](milestone3_guide.md).
Các báo cáo chi tiết nằm trong `reports/milestone3/`, mô hình ở
`artifacts/milestone3/model.joblib`; các đầu ra này được Git ignore và có thể sinh lại.
Bản tổng hợp đã đưa vào Git tại [results/](results/README.md), gồm bảng, bốn hình và hash;
xem [portfolio report](portfolio_report.md) cho bản trình bày hoàn chỉnh của Mốc 4.

## Thiết kế và lựa chọn đã chốt trước test

- Chia ngoài theo thứ tự dòng: train 28.831, validation 6.178, test 6.179.
- Loại duration trong mọi thí nghiệm.
- Ba nhóm biến, ba mô hình, ba fold TimeSeriesSplit chỉ trong train.
- 27 cấu hình mô hình/nhóm biến, tương ứng 81 lần fit theo fold, sau đó refit
  cấu hình tốt nhất của mỗi mô hình/nhóm biến trên toàn bộ train.
- Chọn mô hình bằng AP validation; chọn raw hoặc temporal sigmoid bằng Brier validation;
  chọn ngưỡng bằng utility mô phỏng của kịch bản chính 1:10 và ngân sách validation 20%.
- Baseline mốc 2 được fit lại với cùng train/validation để làm tham chiếu.
- `selection.json` ghi SHA-256 dữ liệu, config, mô hình và báo cáo validation trước lệnh test.
  Test chỉ chấm mô hình đã lưu; không đổi mô hình/ngưỡng từ kết quả test.

| Fold trong train | Số dòng train | Số dòng validation fold | Tỷ lệ yes validation fold |
| --- | ---: | ---: | ---: |
| 1 | 7.210 | 7.207 | 4,75% |
| 2 | 14.417 | 7.207 | 6,41% |
| 3 | 21.624 | 7.207 | 8,19% |

Các fold là khối dòng, không phải khoảng thời gian có độ dài bằng nhau.
Tỷ lệ yes train ngoài là 5,57%, validation ngoài là 10,65%.

## Tuning và độ nhạy theo nhóm biến

Các metric dưới đây dùng score raw. F1 tính tại ngưỡng tối đa F1 riêng của từng ứng viên.
Precision/lift top 20% tính từ xếp hạng score, độc lập với ngưỡng phân loại.

| Nhóm biến / mô hình | AP validation | F1 | Precision top 20% | Lift top 20% |
| --- | ---: | ---: | ---: | ---: |
| Baseline / Dummy | 0,1065 | 0,1925 | 12,14% | 1,139 |
| Baseline / Logistic Regression | 0,1902 | 0,2821 | 20,95% | 1,967 |
| Baseline / Random Forest | 0,1718 | 0,2371 | 17,23% | 1,618 |
| scheduled_assumptions / Logistic Regression | **0,2077** | 0,2857 | 21,12% | 1,983 |
| scheduled_assumptions / Random Forest | 0,1252 | 0,1952 | 11,33% | 1,063 |
| without_campaign / Logistic Regression | 0,1870 | 0,2807 | 20,87% | 1,960 |
| without_campaign / Random Forest | 0,1392 | 0,2011 | 13,51% | 1,269 |
| customer_history_only / Logistic Regression | 0,1973 | 0,2491 | 18,20% | 1,709 |
| customer_history_only / Random Forest | 0,1913 | 0,2513 | 18,93% | 1,778 |

Mô hình được chọn là Logistic Regression của nhóm `scheduled_assumptions`,
với `C=0.1`, `class_weight=None`. AP tăng 0,0175 so với baseline trên cùng validation;
precision top 20% chỉ tăng khoảng 0,16 điểm phần trăm. Chưa có khoảng tin cậy để
kết luận khác biệt có ý nghĩa thống kê.

Giữ campaign cho kết quả cao hơn nhóm bỏ campaign trong thí nghiệm này, nhưng mỗi
nhóm còn có tham số được tuning riêng. Không diễn giải chênh lệch như tác động riêng
của campaign, và không dùng hiệu năng để chứng minh biến có sẵn trước cuộc gọi.
Nhóm chỉ hồ sơ/lịch sử vẫn có AP 0,1973, giúp thấy một phương án với ít giả định hơn.

Random Forest có đầy đủ nhóm biến đạt CV AP cao nhất tại `max_depth=6`, không weighting,
nhưng AP validation chỉ 0,1252, thấp hơn baseline RF 0,1718. Tuning không bảo đảm cải thiện
khi giai đoạn validation khác các fold train. Không thay tham số bằng cách nhìn test.

Mất cân bằng nhãn được kiểm soát qua đối chứng Dummy và thử hai class_weight.
Với Logistic Regression đầy đủ biến ở C=0.1, CV AP không weighting là 0,09448,
balanced là 0,09377; đây là chênh lệch nhỏ. Không kết luận weighting luôn có hại:
ở C=1, balanced đạt 0,09380 so với 0,09095 của không weighting.
Dummy có score bằng nhau; lift của top các dòng hòa điểm không phản ánh khả năng xếp hạng.

## Calibration

| Dạng score của ứng viên được chọn | Brier score | Log loss | AP validation |
| --- | ---: | ---: | ---: |
| Raw | **0,09185** | **0,32381** | **0,20773** |
| Temporal sigmoid ensemble | 0,09531 | 0,33991 | 0,10287 |

Giữ raw vì Brier validation thấp hơn. Sigmoid ensemble không cải thiện trong lần thử này;
khác biệt bao gồm cả việc ensemble các mô hình fold. Không coi calibration luôn có lợi.
Biểu đồ và số mẫu bin trong `calibration.png` và `calibration_bins.csv`.

Bin raw score 0,2–0,3 có 241 mẫu, score trung bình 0,2215 và tỷ lệ yes 0,3071,
cho thấy sự lệch trong nhóm này. Bin 0,3–0,4 chỉ có ba mẫu nên không đủ để kết luận.
Score raw trung bình toàn validation là 0,1162 so với tỷ lệ yes 0,1065;
gần nhau ở trung bình chưa chứng minh calibration tốt ở mọi nhóm.

## Ngưỡng theo kịch bản chi phí

Theo lựa chọn của người dùng, tất cả chi phí/giá trị là minh họa, cùng đơn vị quy ước.
Utility dùng nhãn lịch sử; không đo lợi nhuận nhân quả hoặc doanh thu tăng thêm.

| Chi phí : giá trị mỗi yes | Ngưỡng | Số người liên hệ | Số yes trong nhóm | Utility mô phỏng |
| --- | ---: | ---: | ---: | ---: |
| 1 : 5 | 0,16727 | 766 | 192 | 194 |
| **1 : 10 (chính)** | **0,15314** | **1.064** | **246** | **1.396** |
| 1 : 20 | 0,14695 | 1.233 | 261 | 3.987 |

Ngân sách validation tối đa là 1.236 người. Ngưỡng chính chọn 17,22% validation;
precision 23,12%, recall 37,39%, F1 0,2857, FP 818 và FN 412.
Trong lần chạy này, ngưỡng kinh tế 1:10 trùng ngưỡng tối đa F1. Các kịch bản 1:5 và
1:20 cho ngưỡng khác, nên không suy ra hai quy tắc luôn tương đương.

## Nhóm sai và thay đổi theo thứ tự thời gian

Ở ngưỡng đã chọn, recall khác nhau rõ giữa các nhóm validation:

| Nhóm | Số mẫu | Số yes | Recall | Số FN |
| --- | ---: | ---: | ---: | ---: |
| month=apr | 1.599 | 301 | 75,75% | 73 |
| month=may | 4.579 | 357 | 5,04% | 339 |
| job=admin. | 1.362 | 173 | 49,13% | 88 |
| job=blue-collar | 2.003 | 147 | 16,33% | 123 |

Đây là mô tả lỗi, không chứng minh quan hệ nhân quả hay đánh giá công bằng đã hoàn chỉnh.
Tháng trong dữ liệu không có năm đầy đủ; nhiều nhóm có thể thay đổi cùng các chỉ báo khác.
Nhóm dưới 100 mẫu được đánh dấu trong CSV để tránh diễn giải quá mức.

Chia validation thành năm khối dòng liên tiếp:

| Khối | Tỷ lệ yes | Recall tại ngưỡng cố định | Lift top 20% trong khối |
| --- | ---: | ---: | ---: |
| 1 | 12,14% | 74,00% | 1,661 |
| 2 | 20,31% | 49,00% | 2,125 |
| 3 | 5,74% | 7,04% | 1,825 |
| 4 | 6,56% | 4,94% | 2,222 |
| 5 | 8,50% | 2,86% | 1,286 |

Recall của ngưỡng cố định giảm mạnh ở các khối sau trong khi lift xếp hạng vẫn trên 1.
Ngưỡng và khả năng xếp hạng có thể thay đổi khác nhau; cần theo dõi theo giai đoạn.
Bằng chứng này chưa đủ xác định label shift hay concept drift.

Permutation importance trên validation: pdays giảm AP trung bình 0,04283 khi hoán vị,
month 0,03410, job 0,00884, education 0,00716, campaign 0,00589.
Độ lệch chuẩn qua năm lần nằm trong CSV và hình. Các biến tương quan có thể che tầm
quan trọng nhau; kết quả không phải giải thích nhân quả. pdays=999 vẫn là sentinel
giữ theo baseline, nên cần thử cách biểu diễn khác trong một thí nghiệm riêng sau này.

## Test với mô hình/ngưỡng đã cố định

Test này đã được xem ở baseline/EDA; không phải xác nhận độc lập chưa từng xem.
Không dùng số liệu sau đây để thay đổi lựa chọn ở trên.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Tỷ lệ yes nền | 10,65% | 38,45% |
| AP | 0,2077 | 0,4508 |
| Precision tại ngưỡng 0,15314 | 23,12% | 45,21% |
| Recall tại ngưỡng 0,15314 | 37,39% | 58,46% |
| F1 | 0,2857 | 0,5099 |
| Precision top 20% | 21,12% | 44,90% |
| Lift top 20% | 1,983 | 1,168 |
| Brier score | 0,09185 | 0,27497 |

AP test cao hơn không chứng minh khả năng tổng quát hóa tốt hơn vì tỷ lệ yes nền cũng
tăng mạnh. Lift top 20% giảm xuống 1,168; mức lợi thế so với chọn ngẫu nhiên nhỏ hơn.
Score test trung bình chỉ 19,05% so với tỷ lệ yes 38,45%, và Brier score xấu hơn;
chưa thể diễn giải score thành xác suất đáng tin ở giai đoạn này.

Ngưỡng cố định trên test chọn 3.072/6.179 người (49,72%), vượt tỷ lệ ngân sách 20%
đã ràng buộc trên validation. Con số utility test 10.818 là mô phỏng của ngưỡng cố định
này, không phải kết quả của một chính sách luôn giới hạn 20% cuộc gọi. Muốn ngân sách
vận hành cố định cần chính sách giới hạn theo đợt và đánh giá riêng trên dữ liệu mới.

## Kiểm tra và giới hạn còn lại

- Chín kiểm tra tự động đã qua, gồm thuật toán ngưỡng đối chiếu vét cạn,
  không tách điểm bằng nhau, no-call, thứ tự fold/gap, nhóm một nhãn,
  train-only tuning và test không refit.
- Đã chạy đủ thí nghiệm trên dữ liệu thật, chấm mô hình đóng băng trên test,
  kiểm tra trực quan bốn biểu đồ và thử dự đoán CSV giả lập với ba khách hàng.
- Cấu hình chi phí là minh họa; chưa có chi phí/lợi nhuận thực tế.
- Tính sẵn có trước cuộc gọi của campaign, lịch/kênh và chỉ báo kinh tế chưa xác minh.
- Không có customer ID hoặc ngày đầy đủ để loại trừ người lặp/độ trễ nhãn.
- Validation được dùng lại cho nhiều quyết định; chưa có xác nhận độc lập mới.
- Dữ liệu lịch sử ở Bồ Đào Nha không chứng minh hiệu quả ở chiến dịch/thị trường khác.

Mốc 3 đã thực hiện đủ các thí nghiệm trong kế hoạch ở phạm vi học tập. Mốc 4 đã đóng gói
các kết quả đo và giới hạn này trong báo cáo portfolio, README và model card.
Không quảng bá mô hình như một hệ thống đã xác minh hiệu quả kinh doanh.
