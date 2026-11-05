# Báo cáo thực hành: KNN và mô hình xác suất

## 1. Tóm tắt

Thực nghiệm trên Rice Cammeo/Osmancik cho thấy KNN đã scale với Euclidean, uniform voting và `k=31` đạt test macro-F1 `0.9084`. Gaussian classifier full covariance theo lớp, chọn với scaling và regularization `0.1`, đạt macro-F1 `0.9086` và log loss `0.2026`; Gaussian Naive Bayes đạt macro-F1 `0.8919` và log loss `0.5542`. Trong lần chạy này full covariance tốt hơn trên cả macro-F1 lẫn log loss, nhưng đây là kết quả trên một split của một dataset.

## 2. Dữ liệu và thiết kế thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Dataset | Rice Cammeo/Osmancik |
| Nguồn | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik); ARFF đi kèm ghi Murat Koklu dataset page và trích dẫn nghiên cứu gốc |
| Số mẫu / đặc trưng / lớp | 3.810 / 7 đặc trưng số / 2 lớp |
| Số lượng nhãn | Cammeo: 1.630; Osmancik: 2.180 |
| Thiếu / trùng | 0 giá trị thiếu; 0 hàng trùng |
| Seed và phân hoạch | `SEED=42`, stratified 80/20: 3.048 train, 762 test |
| Chọn cấu hình | Stratified 5-fold CV trên train; metric chính macro-F1 |
| Môi trường thực thi | `ai-algorithms`, Python 3.12.13, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.8.0, Matplotlib 3.10.9 |

Scaling được đặt trong pipeline và fit lại ở từng fold. Test được giữ đến khi lựa chọn cấu hình hoàn tất. Kết quả dưới đây lấy từ hai notebook đã chạy bằng `nbconvert --execute`.

## 3. Bài 1 — K-Nearest Neighbors

### 3.1 Phương pháp

KNN được cài bằng NumPy với Euclidean và Manhattan, uniform/distance weighting. Hàng xóm cách đều được sắp ổn định theo thứ tự training; hòa phiếu chọn lớp đứng trước theo thứ tự lớp; khi có khoảng cách 0, chỉ các láng giềng trùng điểm chia phiếu. CV khảo sát cả scaling bật/tắt, hai metric, hai weighting và `k` từ 1 đến 31.

### 3.2 Kết quả chọn mô hình

Cấu hình tốt nhất theo CV macro-F1 là scaling bật, Euclidean, uniform voting, `k=31`: macro-F1 trung bình `0.9286` (độ lệch chuẩn `0.0113`), accuracy trung bình `0.9301`. Cấu hình được chọn chỉ dựa trên CV.

### 3.3 Đánh giá và giải thích

| Accuracy | Macro-F1 | Confusion matrix (actual × predicted) |
|---:|---:|---|
| 0.9108 | 0.9084 | `[[286, 40], [28, 408]]` theo thứ tự Cammeo, Osmancik |

Mẫu held-out đầu tiên có nhãn thật và dự đoán là Cammeo. Trong 31 láng giềng gần nhất có 30 Cammeo và 1 Osmancik; xác suất phiếu KNN là Cammeo `0.9677`, Osmancik `0.0323`.

Một giới hạn quan sát được là KNN nhạy với scale: đặc trưng hình thái có đơn vị và độ lớn khác nhau nên khoảng cách trước scaling có thể bị chi phối bởi các chiều có trị số lớn. Vì thế việc lựa chọn scaling được đưa vào CV thay vì mặc định từ trước. KNN cũng cần lưu mẫu train để tính láng giềng khi dự đoán.

### 3.4 Đối chiếu scikit-learn

Trên fold validation đầu tiên của training split, `k=7`, Euclidean và uniform voting, NumPy KNN khớp hoàn toàn dự đoán `KNeighborsClassifier` (agreement `1.0000` trên 610 mẫu). Tập test không được dùng ở bước đối chiếu này.

## 4. Bài 2 — Gaussian probabilistic models

### 4.1 Phương pháp

Gaussian Naive Bayes ước lượng prior, mean và phương sai MLE theo lớp bằng NumPy; phương sai được cộng epsilon theo `var_smoothing`, log posterior được chuẩn hóa bằng log-sum-exp. Gaussian full covariance ước lượng covariance MLE riêng từng lớp và cộng regularization đường chéo. Mô hình được so sánh qua macro-F1 CV; cấu hình được chọn riêng trong từng họ mô hình trước khi đánh giá test.

### 4.2 Lựa chọn mô hình và kết quả test

CV macro-F1 cao nhất cho full covariance là `0.93172` (độ lệch chuẩn `0.00729`), scaling bật, regularization `0.1`. Gaussian Naive Bayes tốt nhất đạt CV macro-F1 `0.92123` (độ lệch chuẩn `0.00912`), scaling bật, `var_smoothing=1e-9`.

| Mô hình | Cấu hình đã chốt | Accuracy | Macro-F1 | Log loss |
|---|---|---:|---:|---:|
| Gaussian Naive Bayes | scaling; `var_smoothing=1e-9` | 0.8950 | 0.8919 | 0.5542 |
| Gaussian full covariance | scaling; `reg_covar=0.1` | 0.9108 | 0.9086 | 0.2026 |

Confusion matrix theo thứ tự Cammeo, Osmancik:

- Gaussian Naive Bayes: `[[276, 50], [30, 406]]`.
- Gaussian full covariance: `[[288, 38], [30, 406]]`.

### 4.3 Diễn giải và thảo luận

Với held-out sample đầu tiên, full covariance dự đoán Cammeo. Prior Cammeo/Osmancik lần lượt là `0.4278`/`0.5722`; log-likelihood lần lượt là `-2.6023`/`-7.6670`; posterior sau chuẩn hóa là `0.9916`/`0.0084`.

Full covariance cho macro-F1 cao hơn khoảng `0.0167` và log loss thấp hơn khoảng `0.3516` so với Gaussian Naive Bayes trong split này. Trên training split, tương quan tuyệt đối giữa `Area` và `Convex_Area` trong từng lớp là `0.9974` (Cammeo) và `0.9978` (Osmancik), cho thấy giả định độc lập có điều kiện của Naive Bayes chỉ là xấp xỉ. Full covariance có thể biểu diễn quan hệ giữa các đặc trưng hình thái; đây là diễn giải trên một split, không chứng minh tương quan là nguyên nhân duy nhất của chênh lệch điểm số.

### 4.4 Đối chiếu scikit-learn

Với `GaussianNB(var_smoothing=1e-9)` trên fold validation đầu tiên của training split, dự đoán khớp hoàn toàn. Sai khác xác suất lớn nhất là `3.55e-15`, phù hợp sai số dấu phẩy động. Tập test không được dùng ở bước đối chiếu này.

## 5. Hạn chế và kết luận

- Kết quả phụ thuộc vào một train/test split với seed cố định; CV giảm phụ thuộc vào một validation split nhưng không thay thế đánh giá trên dataset khác.
- KNN đạt kết quả cạnh tranh nhưng cần scaling và lưu dữ liệu train.
- Full covariance tốt hơn Naive Bayes theo ba metric test ở lần chạy này; mô hình có nhiều tham số covariance hơn và regularization được chọn bằng CV.
- Không suy rộng kết quả sang dataset hoặc phân phối dữ liệu khác.

## 6. Khả năng tái lập

- **Notebook KNN:** `problem1_knn_numpy.ipynb` — đã chạy từ đầu đến cuối, không có cell lỗi.
- **Notebook Gaussian:** `problem2_gaussian_models.ipynb` — đã chạy từ đầu đến cuối, không có cell lỗi.
- **Lệnh thực thi:** `conda run -n ai-algorithms jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <notebook>`.
- **Dữ liệu:** `data/Rice_Cammeo_Osmancik.arff`; generator notebook: `scripts/build_notebooks.py`.
- **Seed:** 42.
