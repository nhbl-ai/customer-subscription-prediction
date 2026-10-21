# Chuẩn bị public và đưa vào CV

Repo đã có bốn mốc portfolio, README tiếng Anh, báo cáo tiếng Việt, CHANGELOG,
MIT license cho mã nguồn và attribution riêng cho dữ liệu UCI (CC BY 4.0).
Phiên bản package chuẩn bị là 1.0.0; chưa đồng nghĩa có tag/release trên GitHub.

## Kết quả kiểm tra cục bộ ngày 21/10/2026

- 10/10 kiểm thử qua trên Python 3.13.9/Windows, gồm export và kiểm tra provenance.
- Build và cài wheel 1.0.0 thành công; chạy lại cả 10 kiểm thử và CLI trên bản sao
  chỉ chứa file dành cho repo, dùng package từ wheel, không có CSV UCI/model sẵn.
- Demo inference trên model thật với CSV giả lập xuất được danh sách top 2.
- `pip check`, `git diff --check`, link Markdown cục bộ và schema notebook đều qua;
  notebook có 10 code cell đã thực thi, không có output lỗi.
- Hash của 17 file bằng chứng khớp manifest và giữ nguyên qua bộ lọc nội dung của Git.
- Quét mẫu secret phổ biến trong file công khai và lịch sử Git cục bộ không có kết quả;
  các file chuẩn bị công khai không còn đường dẫn máy cá nhân.

Đây là kiểm tra cục bộ; chưa xác nhận matrix CI từ xa hoặc đổi visibility GitHub.

## Kiểm tra bản cục bộ

Tại thư mục gốc, dùng Python trong môi trường đã cài dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip wheel . --no-deps --wheel-dir dist
git diff --check
git status --short
```

Thư mục data/raw, data/processed, artifacts, reports, .venv và file .env được ignore.
Kết quả tổng hợp công khai nằm trong docs/results. Không dùng `git add -f` để thêm
dữ liệu hoặc model; chỉ CSV giả lập trong data/examples được đưa vào repo.

## Đưa thay đổi lên GitHub

Review các thay đổi, kể cả mã Mốc 1–3 đã có trong workspace trước phần đóng gói:

```powershell
git diff
git ls-files --others --exclude-standard
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Complete ML portfolio milestones and prepare public v1.0.0"
git push origin HEAD
```

Chờ workflow **CI** hoàn tất trên GitHub. Matrix cấu hình Python 3.11/3.13,
Ubuntu/Windows; không mô tả là đã pass từ xa cho đến khi thấy kết quả thực tế.

Khi đã sẵn sàng công khai, trong GitHub repository **Settings → General → Danger Zone →
Change repository visibility → Public**. Mở repo trong cửa sổ riêng tư để kiểm tra
README, bốn hình, notebook và link đều đọc được mà không cần đăng nhập.
Việc chuẩn bị file cục bộ không tự đổi visibility GitHub.

Có thể tạo tag/release `v1.0.0` sau khi CI pass; nội dung lấy từ CHANGELOG.
Không cần upload model hoặc toàn bộ CSV để người tuyển dụng đánh giá mã nguồn.

## Hoàn thiện trang repo và CV

Suggested About description:

> Bank subscription ranking with temporal validation, leakage-aware preprocessing, calibration diagnostics and reproducible scikit-learn experiments.

Suggested topics: `machine-learning`, `scikit-learn`, `bank-marketing`, `classification`,
`temporal-validation`, `data-science`, `portfolio`.

Link CV: [github.com/nhbl-ai/customer-subscription-prediction](https://github.com/nhbl-ai/customer-subscription-prediction).
Các bullet tiếng Anh có sẵn cuối [portfolio report](portfolio_report.md).
Chỉ dùng kết quả offline đã đo; khi nói lift 1,98× phải ghi là validation top 20%.
