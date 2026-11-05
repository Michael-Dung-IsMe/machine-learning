# Bài tập môn Học máy (INT14121_CLC)

## Mục đích repo

Đây là danh sách các bài thực hành thuộc môn Học máy (Machine Learning). Mỗi thư mục bài tập có tài liệu riêng để giải thích phương pháp, cách chạy và kết quả thực nghiệm.

## Cấu trúc repo

```text
machine-learning/
├── linear-regression/    # Hồi quy tuyến tính và các phương pháp cải thiện
├── logistic-regression/  # Phân loại bằng hồi quy logistic
├── knn-prob-model/       # KNN và các mô hình xác suất Gaussian
├── random-forest/        # Decision Trees, Random Forest và Gradient Boosting
├── svm/                  # SVM tuyến tính và SVM dùng kernel
├── environment.yml       # Khai báo môi trường Conda dùng chung
└── README.md             # Tổng quan repo và hướng dẫn cài đặt
```

Trong mỗi thư mục bài tập, mở `README.md` để xem cấu trúc chi tiết, dữ liệu và cách chạy của bài đó.

## Danh sách bài tập đã upload

Thời gian là ngày bài tập được đưa vào Git lần đầu. Bảng chỉ liệt kê các thư mục hiện có trên `origin/master`.

| Tên bài | Thời gian | Nội dung chính | Đường dẫn tương đối |
|---|---|---|---|
| Linear Regression | 2026-08-26 | Dự đoán số lượt thuê xe đạp; cài đặt hồi quy tuyến tính, mở rộng đặc trưng và regularization. | [linear-regression/](linear-regression/README.md) |
| Logistic Regression | 2026-11-05 | Phân loại giống lúa mì Seeds; cài đặt logistic regression bằng NumPy, so sánh OvR với multinomial và khảo sát ngưỡng phân loại. | [logistic-regression/](logistic-regression/README.md) |
| KNN & Probabilistic Models | 2026-11-05 | Phân loại giống gạo Cammeo/Osmancik; cài đặt KNN, Gaussian Naive Bayes và Gaussian full covariance, rồi so sánh kết quả. | [knn-prob-model/](knn-prob-model/README.md) |
| Decision Trees, Random Forest & Gradient Boosting | 2026-11-05 | Phân loại giống gạo; khảo sát cây quyết định, cắt tỉa, bagging, Random Forest và Gradient Boosting. | [random-forest/](random-forest/README.md) |
| Support Vector Machine (SVM) | 2026-11-05 | Phân loại giống gạo; so sánh SVM tuyến tính với kernel polynomial/RBF, scaling và lựa chọn siêu tham số. | [svm/](svm/README.md) |

## Yêu cầu phiên bản và môi trường

- Cần cài Conda hoặc Miniconda để tạo môi trường từ [`environment.yml`](environment.yml).
- Môi trường có tên `ml-implementation`, dùng Python `>=3.9,<3.12` (tức Python 3.9–3.11).
- Các thư viện chính được khai báo trong file: NumPy `>=1.23`, pandas `>=1.5`, SciPy `>=1.10`, scikit-learn `>=1.2`, Matplotlib `>=3.6`, JupyterLab `>=4.0`, ipykernel `>=6.20` và pytest `>=7.0`.

Lưu ý: báo cáo của bài KNN ghi nhận kết quả từ một môi trường Python 3.12 riêng; chạy bằng `environment.yml` có thể cho phiên bản thư viện và kết quả khác đôi chút.

## Hướng dẫn cài đặt

Tại thư mục gốc của repo, chạy:

```bash
conda env create -f environment.yml
conda activate ml-implementation
jupyter lab
```

Trong JupyterLab, mở notebook của bài tập cần thực hành và chọn kernel của môi trường `ml-implementation`. Nếu môi trường đã được tạo trước đó và file cấu hình thay đổi, cập nhật bằng `conda env update -f environment.yml --prune`.
