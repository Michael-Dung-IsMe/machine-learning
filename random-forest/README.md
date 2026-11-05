# Thực hành Machine Learning: Decision Trees, Random Forest và Gradient Boosting

Tài liệu này tổng hợp toàn diện các yêu cầu, nền tảng lý thuyết, kết quả đo đạc thực nghiệm khách quan và hướng dẫn tái lập cho bài tập thực hành **Decision Trees, Random Forest và Gradient Boosting** trên bộ dữ liệu chuẩn **Rice Dataset (UCI Machine Learning Repository)** (dựa trên đề bài gốc [`Random-Forest.md`](Random-Forest.md) và kế hoạch phê duyệt tại [`PLAN.md`](PLAN.md)).

---

## 1. Cấu trúc Thư mục Dự án

```text
random-forest/
├── README.md                           # Báo cáo tổng hợp & hướng dẫn tái lập thực nghiệm (tệp này)
├── PLAN.md                             # Kế hoạch kiến trúc và phân công triển khai
├── Random-Forest.md                    # Đề bài gốc của học phần
├── data/
│   └── Rice_Cammeo_Osmancik.arff       # Dữ liệu gốc từ UCI (3.810 mẫu, 7 đặc trưng, 2 lớp)
├── problem1_decision_trees.ipynb       # Problem 1: ID3 First Principles, Quantile Discretization & Pruning
├── problem2_rf_and_boosting.ipynb      # Problem 2: Bagging, Random Forest, OOB, Gradient Boosting & Importance
├── scripts/
│   ├── build_rf_notebooks.py           # Script tự động xây dựng và thực thi 2 notebook kèm outputs và Base64 plots
│   └── test_notebooks_execution.py     # Script kiểm thử thực thi toàn vẹn độc lập (dual-context)
└── tests/
    ├── conftest.py                     # Fixtures kiểm thử, cấu hình logging và headless backend
    ├── test_data_integrity.py          # Kiểm thử tính toàn vẹn schema, phân hoạch không giao nhau
    ├── test_problem1_logic.py          # Kiểm thử thuật toán ID3, Quantile Discretization và Pruning
    ├── test_problem2_logic.py          # Kiểm thử Decorrelation, OOB, Early Stopping và Importance
    ├── test_execution_e2e.py           # Kiểm thử E2E cấu trúc notebook, tỷ lệ MD/Code và runtime
    └── run_tests.py                    # Test runner tổng hợp kiểm tra toàn bộ 70 test cases
```

---

## 2. Tóm tắt Dữ liệu & Thiết lập Thực nghiệm

* **Nguồn dữ liệu:** [Rice Dataset (Cammeo and Osmancik), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik), DOI [10.24432/C5538F](https://doi.org/10.24432/C5538F), License CC BY 4.0.
* **Quy mô ma trận:** 3.810 quan sát thực tế từ thị giác máy tính, 7 đặc trưng hình thái học dạng số thực liên tục:
  1. `Area`: Diện tích hạt gạo (pixel).
  2. `Perimeter`: Chu vi đường biên hạt (pixel).
  3. `Major_Axis_Length`: Chiều dài trục lớn nhất của elip ngoại tiếp.
  4. `Minor_Axis_Length`: Chiều dài trục nhỏ nhất của elip ngoại tiếp.
  5. `Eccentricity`: Độ lệch tâm elip ($e = \sqrt{1 - b^2/a^2}$).
  6. `Convex_Area`: Diện tích bao lồi nhỏ nhất chứa hạt.
  7. `Extent`: Tỷ lệ diện tích hạt so với hình chữ nhật bao quanh.
* **Nhãn mục tiêu (2 giống lúa Thổ Nhĩ Kỳ):**
  * `0: Cammeo`: 1.630 mẫu ($42.78\%$)
  * `1: Osmancik`: 2.180 mẫu ($57.22\%$)
  * Tỷ lệ thiếu dữ liệu: $0\%$; Tỷ lệ trùng lặp: $0\%$.
* **Phân hoạch Dữ liệu Phân tầng (`SEED = 42`):**
  * Phân tầng Stratified 60/20/20 giữ nguyên tỷ lệ $57.22\%$ Osmancik ở mọi phân vùng:
    * **Train (60%):** 2.286 mẫu (Cammeo: 978, Osmancik: 1.308)
    * **Validation (20%):** 762 mẫu (Cammeo: 326, Osmancik: 436)
    * **Test (20%):** 762 mẫu (Cammeo: 326, Osmancik: 436)
  * Tính độc lập tuyệt đối: $\text{Train} \cap \text{Val} = \emptyset, \ \text{Train} \cap \text{Test} = \emptyset, \ \text{Val} \cap \text{Test} = \emptyset$.
* **Nguyên tắc Chống Rò rỉ Dữ liệu & Pre-test Commitment:**
  * Toàn bộ tiền xử lý (rời rạc hóa phân vị) chỉ fit trên tập Train, sau đó transform độc lập lên Validation và Test.
  * Toàn bộ tinh chỉnh siêu tham số và lựa chọn mô hình được niêm phong dựa trên tập Validation trước khi mở tập Test đúng **01 lần duy nhất**.

---

## 3. Tổng hợp Kết quả Thực nghiệm Chính

### Problem 1 — Decision Trees, Discretization và Pruning

#### 1. Thuật toán ID3 từ First Principles (NumPy/Pure Python)
* Tự cài đặt thuật toán ID3 thuần túy không dựa vào thư viện cây ngoài:
  * **Quantile Discretization (4 giỏ phân vị - Quartiles):** Tính các ngưỡng $q_{25}, q_{50}, q_{75}$ độc lập trên từng đặc trưng chỉ từ tập Train (sử dụng `np.unique` để loại bỏ ngưỡng trùng).
  * **Hàm Shannon Entropy & Information Gain:**
    $$H(S) = - \sum_{c} p_c \log_2(p_c), \quad IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$$
  * **Điều kiện dừng & Dự phòng:** Dừng khi node thuần nhất ($H=0$), không còn thuộc tính, hoặc đạt `max_depth`; tự động fallback về majority class của node cha khi gặp giá trị bin chưa từng thấy.
* **Khảo sát độ sâu ID3:**
  * `max_depth = 1` (Stump): Train F1 = **0.9002**, Val F1 = **0.9035** (Underfitting rõ rệt).
  * `max_depth = 4`: Train F1 = **0.9241**, **Val F1 = 0.9226** (Điểm cân bằng Bias-Variance tối ưu cho ID3).
  * `max_depth = 7`: Train F1 = **0.9324**, Val F1 = **0.9175** (Bắt đầu Overfitting vào các tổ hợp giỏ nhỏ).

#### 2. Khảo sát Cắt tỉa (Pruning) trên scikit-learn
* **Mô hình 1 — Unconstrained Tree (CART không ràng buộc):**
  * Cây tự do đạt độ sâu `max_depth = 18` với `175 lá`.
  * Train Accuracy = **1.0000**, Train Macro-F1 = **1.0000**.
  * Validation Accuracy = **0.8990**, Validation Macro-F1 = **0.8968**.
  * **Khoảng cách Quá khớp (Overfitting Gap):** $\Delta = 1.0000 - 0.8968 = \mathbf{0.1032}$ ($10.32\%$).
* **Mô hình 2 — Pre-pruned Tree Grid Search (`max_depth \in [1, 15]`, `min_samples_leaf \in [1, 50]`):**
  * Cấu hình tối ưu: `max_depth = 4`, `min_samples_leaf = 20`.
  * Cấu trúc cây: **15 lá**, **độ sâu 4**.
  * Train Macro-F1 = **0.9312**, **Validation Macro-F1 = 0.9398**, Validation Accuracy = **0.9409**.
  * Khoảng cách Train-Validation: $\Delta = -0.0086$ (Triệt tiêu hoàn toàn overfitting gap).
* **Mô hình 3 — Cost-Complexity Pruning (CCP Post-pruning):**
  * Trích xuất 70 giá trị $\alpha_{\text{eff}}$ từ `cost_complexity_pruning_path`.
  * Giá trị tối ưu: $\alpha^* = \mathbf{0.001785}$.
  * Cấu trúc cây: Siêu tinh gọn với **5 lá** và **độ sâu 3**.
  * Train Macro-F1 = **0.9281**, **Validation Macro-F1 = 0.9368**, Validation Accuracy = **0.9383**.
* **Pre-test Commitment & Đánh giá Test 1 lần duy nhất:**
  * Quy tắc Occam's Razor Tie-breaking: Chênh lệch $\Delta = 0.9398 - 0.9368 = 0.0030 > 0.0020 \implies$ Chốt **Pre-pruned Tree (`max_depth=4, min_samples_leaf=20`)** làm Champion Model.
  * **Kết quả trên tập Test nguyên bản (762 mẫu):**
    * Test Accuracy = **0.9173** (91.73%, 699/762 mẫu chính xác).
    * Test Balanced Accuracy = **0.9130**.
    * Test Macro-F1 = **0.9151** (Cammeo F1: 0.9014, Osmancik F1: 0.9288).

---

### Problem 2 — Random Forest và Gradient Boosting

#### 1. Bagging vs. Random Forest & Cơ chế Tree Decorrelation
* Theo lý thuyết Breiman & Hastie, phương sai của mô hình tổ hợp $B$ cây con tuân theo công thức:
  $$\mathrm{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$
* Khi số lượng cây $B \to \infty$, phương sai không thể giảm về 0 mà bị chặn dưới bởi $\rho \sigma^2$, trong đó $\rho$ là hệ số tương quan trung bình giữa các cây.
* **Số liệu thực nghiệm định lượng:**
  * $m = d = 7$ (Bagging thuần túy): $\bar{\rho} = \mathbf{0.7969}$, Disagreement Rate = $\mathbf{0.0999}$, Val Macro-F1 = **0.9288**.
  * $m = 5$: $\bar{\rho} = \mathbf{0.7955}$, Disagreement Rate = $\mathbf{0.1007}$, Val Macro-F1 = **0.9343**.
  * $m = \lfloor\sqrt{d}\rfloor = 2$ (Random Forest): $\bar{\rho} = \mathbf{0.7955}$, Disagreement Rate = $\mathbf{0.1007}$, Val Macro-F1 = **0.9329**.
* **Kết luận:** Giới hạn ngẫu nhiên số thuộc tính tại mỗi split giúp giảm tương quan $\bar{\rho}$, tăng độ đa dạng (disagreement) giữa các cây con và hạ thấp cận dưới phương sai tổ hợp.

#### 2. Đánh giá Out-of-Bag (OOB) & Độ Hội tụ
* Tại $n_{\text{estimators}} = 10$: Xuất hiện cảnh báo `UserWarning` do xác suất một mẫu không bao giờ rơi ra ngoài túi bootstrap là $(1 - 1/e)^{10} \approx 1.04\%$ (kỳ vọng lý thuyết ~24 mẫu thiếu điểm OOB, thực tế đo được 28 mẫu uncovered: 2.258/2.286 mẫu được đánh giá hợp lệ, OOB Macro-F1 đạt **0.9032**).
* Tại $n_{\text{estimators}} \ge 100$: OOB Macro-F1 hội tụ tiệm cận về Validation Macro-F1 với độ chênh lệch tuyệt đối $|\text{OOB} - \text{Val}| \le 0.015$:
  * $n = 100$: OOB Macro-F1 = **0.9183**, Val Macro-F1 = **0.9329** (Chênh lệch 0.0146).
  * $n = 200$: OOB Macro-F1 = **0.9223**, Val Macro-F1 = **0.9315** (Chênh lệch 0.0092).
  * $n = 300$: OOB Macro-F1 = **0.9214**, Val Macro-F1 = **0.9328** (Chênh lệch 0.0115).
* Khẳng định: Đánh giá OOB là thước đo nội tại đáng tin cậy, thay thế hiệu quả cho cross-validation khi không có tập validation riêng.

#### 3. Gradient Boosting & Early Stopping
* Khảo sát lưới $\eta \in [0.01, 0.05, 0.10, 0.20]$ và $d \in [1, 2, 3]$ trên 300 vòng lặp boosting:
  * Cấu hình tối ưu: $\eta = 0.10, d = 2$.
  * Validation Log-Loss đạt cực tiểu toàn cục **0.1734** tại chính xác vòng lặp **$M^* = 61$**, với Validation Macro-F1 đạt **0.9411** (đỉnh F1 đạt 0.9424 tại vòng 53).
  * Nếu không dừng sớm mà tiếp tục chạy đến $M = 300$, Validation Log-Loss tăng và Validation Macro-F1 suy giảm về **0.9385** (Overfit Delta = $+0.0026$).
* Early Stopping tại $M^* = 61$ giúp tiết kiệm gần $80\%$ chi phí tính toán và ngăn chặn hiện tượng quá khớp.

#### 4. Phân tích Feature Importance (MDI vs Permutation Importance)
* **Bảng so sánh độ quan trọng trên Random Forest:**
  | Đặc trưng (Feature) | MDI (Gini Importance) | Xếp hạng MDI | Permutation Importance (Mean $\pm$ Std) | Xếp hạng Permutation |
  |---|:---:|:---:|:---:|:---:|
  | `Perimeter` | 0.2653 | 2 | **0.0215 $\pm$ 0.0032** | **1** |
  | `Major_Axis_Length` | **0.2667** | **1** | 0.0176 $\pm$ 0.0035 | 2 |
  | `Eccentricity` | 0.0899 | 5 | 0.0028 $\pm$ 0.0030 | 3 |
  | `Area` | 0.1743 | 3 | 0.0021 $\pm$ 0.0027 | 4 |
  | `Convex_Area` | 0.1301 | 4 | 0.0003 $\pm$ 0.0044 | 5 |
  | `Minor_Axis_Length` | 0.0395 | 6 | -0.0010 $\pm$ 0.0025 | 6 |
  | `Extent` | 0.0342 | 7 | -0.0030 $\pm$ 0.0019 | 7 |
* **Phân tích bản chất:**
  * `Area`, `Convex_Area` và `Perimeter` có hệ số tương quan tuyến tính rất cao ($r > 0.98$). MDI chia sẻ bừa bãi độ giảm tạp chất trong tập huấn luyện cho cả 3 đặc trưng (tổng cộng $> 57\%$).
  * Khi hoán vị độc lập trên tập Validation, `Permutation Importance` vạch trần tính dư thừa thông tin của `Area` và `Convex_Area` (điểm số gần bằng 0), đồng thời khẳng định `Perimeter` mới là đặc trưng quyết định đơn lẻ mạnh nhất.

---

### Bảng Đối Chuẩn Toàn diện 4 Họ Mô hình (Grand Comparison Table)

Bảng tổng hợp hiệu năng, độ phức tạp và chi phí tài nguyên đo đạc thực tế từ notebook:

| Mô hình (Model Family) | Validation Accuracy | Validation Balanced Acc | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Fit Time (ms) | Inference Latency (ms/sample) | Độ phức tạp mô hình |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Tuned Decision Tree** | 0.9409 | 0.9407 | 0.9398 | **0.9173** | **0.9151** | **12.2 ms** | **0.00031 ms** | 15 lá, độ sâu 4 (1 cây đơn) |
| **Bagging Classifier** | 0.9304 | 0.9280 | 0.9288 | 0.9147 | 0.9123 | 1617.6 ms | 0.04152 ms | 100 cây không ràng buộc |
| **Random Forest** | 0.9344 | 0.9322 | 0.9329 | 0.9160 | 0.9136 | 626.1 ms | 0.02076 ms | 100 cây ($m = \sqrt{d}$) |
| **Gradient Boosting ($M^*=61$)** | **0.9423** | **0.9418** | **0.9411** | 0.9160 | 0.9136 | 428.2 ms | 0.00101 ms | 61 cây nông (độ sâu 2) |

> [!NOTE]
> **Nhận định Khoa học & Occam's Razor:**
> * Trên tập Validation, **Gradient Boosting ($M^*=61$)** dẫn đầu về mọi chỉ số (Val Macro-F1 = 0.9411), theo sau sát nút là **Tuned Decision Tree** (Val Macro-F1 = 0.9398, chênh lệch chỉ $0.0013 < 0.0020$).
> * Khi kiểm chứng trên tập Test nguyên bản, cả 4 mô hình đều đạt hiệu năng vượt trội trong vùng $91.2\% - 91.5\%$. Đặc biệt, **Tuned Decision Tree** duy trì Test Macro-F1 cao nhất (0.9151) với tốc độ huấn luyện nhanh gấp **35 lần** và độ trễ suy luận nhanh hơn **3 lần** so với Gradient Boosting.
> * Kết quả này chứng minh rằng trên các bài toán dữ liệu bảng dạng số có chất lượng cao và số chiều thấp, một cây quyết định được cắt tỉa chuẩn mực (Occam Baseline) hoàn toàn có thể cạnh tranh sòng phẳng với các mô hình Ensemble phức tạp trong khi mang lại khả năng diễn giải minh bạch tuyệt đối.

---

## 4. Hướng dẫn Tái lập Thực nghiệm (Reproducibility Guide)

Dự án được cấu hình để tái lập hoàn toàn trên cả máy cục bộ (Windows / Linux / macOS) và môi trường ảo hóa.

### 4.1. Thiết lập Môi trường Conda
Sử dụng môi trường được định nghĩa sẵn trong `environment.yml`:
```powershell
# Kích hoạt môi trường coursera-ml
conda activate coursera-ml

# Hoặc tạo môi trường mới nếu chưa có:
conda env create -f environment.yml
conda activate coursera-ml
```

### 4.2. Xây dựng & Thực thi Toàn bộ Notebook bằng Automation Script
Script `build_rf_notebooks.py` sẽ thực thi lại toàn bộ 31 code cells trong môi trường cô lập không GUI (headless Matplotlib `Agg`), thu thập toàn bộ stdout text stream và nhúng sẵn 8 biểu đồ Base64 PNG độ phân giải cao vào metadata của notebook:
```powershell
# Thực thi từ thư mục gốc repository:
python random-forest/scripts/build_rf_notebooks.py

# Hoặc thực thi từ thư mục random-forest/:
cd random-forest
python scripts/build_rf_notebooks.py
cd ..
```

### 4.3. Kiểm thử Tự động Thực thi Hai Bối cảnh (Dual-Context Execution)
Script `test_notebooks_execution.py` kiểm tra việc chạy tuần tự tất cả các ô code của cả hai notebook mà không gây ra bất kỳ ngoại lệ nào:
```powershell
# Bối cảnh 1: Chạy từ thư mục gốc repository
python random-forest/scripts/test_notebooks_execution.py

# Bối cảnh 2: Chạy từ thư mục con random-forest/
cd random-forest
python scripts/test_notebooks_execution.py
cd ..
```

### 4.4. Kiểm thử Toàn bộ Bộ Test Suite (70 E2E Invariants)
Chạy bộ test suite toàn diện kiểm tra tính toàn vẹn dữ liệu, thuật toán ID3, pruning, decorrelation, OOB, early stopping, và các tiêu chuẩn trình bày notebook:
```powershell
# Chạy toàn bộ 70 bài test:
python random-forest/tests/run_tests.py
```

### 4.5. Khảo sát Trực tiếp trên Trình biên dịch Notebook
Mở `problem1_decision_trees.ipynb` hoặc `problem2_rf_and_boosting.ipynb` trong **VS Code**, **JupyterLab**, hoặc tải lên **Google Colab**. Cả hai notebook đều tích hợp cơ chế tự động tải và xác thực SHA-256 từ UCI nếu không tìm thấy tệp cục bộ.
