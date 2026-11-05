# Thực hành KNN và mô hình xác suất

Tài liệu yêu cầu gốc: [docs/KNN-ProbModel.md](docs/KNN-ProbModel.md). Bài gồm hai bài tập độc lập; mỗi bài chọn một bộ dữ liệu phân loại có đặc trưng số và nộp một notebook chạy được từ đầu đến cuối. Có thể dùng chung bộ dữ liệu cho cả hai bài.

## Đầu việc

### Bài 1 — K-Nearest Neighbors

- Tự cài đặt KNN bằng NumPy; không dùng classifier hay neighbor-search routine của thư viện.
- Hỗ trợ hai phép đo khoảng cách, bỏ phiếu đồng đều và theo khoảng cách; nêu cách xử lý hòa phiếu và khoảng cách bằng 0.
- Đối chiếu dự đoán với scikit-learn.
- Khảo sát scaling, số láng giềng `k`, khoảng cách và weighting; dùng cross-validation chọn cấu hình.
- Đặt preprocessing bên trong training folds, giữ test set cho đánh giá cuối; chọn và giải thích metric chính.
- Vẽ train/validation performance theo `k`; giải thích một dự đoán held-out qua láng giềng và thảo luận một giới hạn quan sát được.

### Bài 2 — Mô hình xác suất

- Tự cài Gaussian Naive Bayes bằng NumPy, không dùng classifier hoặc probability-density routine của thư viện. Ước lượng tham số bằng maximum likelihood (MLE), trả xác suất lớp đã chuẩn hóa, xử lý ổn định số và đối chiếu với scikit-learn.
- So sánh với Gaussian classifier có ma trận hiệp phương sai đầy đủ riêng cho từng lớp.
- Dùng cross-validation chọn mô hình/cấu hình; đặt preprocessing và tuning trong training folds, giữ test set cho đánh giá cuối.
- Giải thích một dự đoán held-out bằng prior, likelihood và posterior.
- Chốt cấu hình trước khi báo cáo accuracy, macro-F1, log loss và confusion matrix trên test.
- Thảo luận giả định độc lập có điều kiện và liệu phân loại tốt hơn có đồng nghĩa xác suất tốt hơn không.

## Quy trình thực hiện

1. Chọn dataset cho mỗi bài từ danh sách trong đề hoặc một dataset phân loại thực có đặc trưng số. Ghi nguồn, mục tiêu, kích thước và kiểm tra dữ liệu.
2. Chốt seed, phân hoạch phân tầng và metric chính; giữ test set chưa chạm đến. Không fit preprocessing ngoài training fold.
3. Xây dựng NumPy, kiểm tra ca biên và đối chiếu với scikit-learn.
4. Chạy thí nghiệm theo từng bài; trình bày bảng/đồ thị và giải thích kết quả.
5. Chọn cấu hình bằng validation/CV, đóng băng cấu hình rồi đánh giá test cuối.
6. Chạy notebook từ kernel sạch; lưu code, output, bảng, đồ thị và giải thích.

## Môi trường và chạy

Từ thư mục gốc repository, dùng môi trường đã khai báo trong `environment.yml`:

```powershell
conda activate ml-implementation
cd knn-prob-model
jupyter lab
```

Chọn kernel của môi trường, mở từng notebook và chạy **Run All**. Không cài thêm dependency riêng nếu các gói hiện có đáp ứng yêu cầu. Notebook cần mô tả cách lấy dữ liệu và chạy độc lập từ thư mục bài hoặc repository root.

## Tài liệu trong thư mục

- [docs/KNN-ProbModel.md](docs/KNN-ProbModel.md) — đề bài gốc và danh sách dataset tham khảo.
- [docs/implementation-plan.md](docs/implementation-plan.md) — kế hoạch theo giai đoạn cho cả hai bài.
- [docs/acceptance-tests.md](docs/acceptance-tests.md) — kiểm tra nghiệm thu và bằng chứng cần có.
- [docs/report.md](docs/report.md) — khung báo cáo riêng để điền kết quả thực nghiệm đã chạy.

Notebook và dữ liệu nằm cạnh README: `problem1_knn_numpy.ipynb`, `problem2_gaussian_models.ipynb`, và `data/Rice_Cammeo_Osmancik.arff`.

## Sản phẩm cần nộp

Hai notebook Jupyter, mỗi notebook tương ứng một bài tập, chạy từ đầu đến cuối và có dữ liệu truy xuất được, seed cố định, mã nguồn, đồ thị/bảng và diễn giải. Báo cáo phải dùng kết quả thực tế từ lần chạy đã ghi nhận; không điền số liệu minh họa như số đo thực nghiệm.

## Dữ liệu và tái tạo notebook

Dữ liệu Rice Cammeo/Osmancik được cung cấp tại `data/Rice_Cammeo_Osmancik.arff`. Thông tin bộ dữ liệu có tại [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik); header ARFF đi kèm ghi Murat Koklu dataset page và trích dẫn bài báo gốc.

Các notebook được tạo từ `scripts/build_notebooks.py`. Sau khi tạo môi trường `ml-implementation` từ `environment.yml`, có thể chạy từng notebook từ thư mục repository root bằng `jupyter nbconvert --to notebook --execute --inplace knn-prob-model/problem1_knn_numpy.ipynb` và lệnh tương ứng cho `problem2_gaussian_models.ipynb`.

Lần xác minh hiện tại dùng môi trường sẵn có `ai-algorithms` (Python 3.12.13, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.8.0, Matplotlib 3.10.9); cả hai notebook đã chạy hoàn tất với seed 42.
