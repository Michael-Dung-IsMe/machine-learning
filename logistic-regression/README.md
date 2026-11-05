# Thực hành Logistic Regression — Seeds Dataset (Kama, Rosa, Canadian)

Tài liệu này tổng hợp toàn diện các yêu cầu, mục tiêu học thuật, kết quả thực nghiệm và hướng dẫn tái lập cho bài thực hành **Logistic Regression** (dựa trên đề bài gốc [`Logistic-Regression.md`](Logistic-Regression.md)).

---

## 1. Cấu trúc Thư mục Dự án

```text
logistic-regression/
├── README.md                                 # Hướng dẫn & tổng hợp kết quả (tài liệu này)
├── Logistic-Regression.md                    # Đề bài gốc
├── data/
│   └── seeds_dataset.txt                     # Dữ liệu hạt giống gốc từ UCI (210 mẫu, 7 đặc trưng, 3 lớp)
├── problem1_first_principles_lr.ipynb        # Problem 1: Tự cài đặt Logistic Regression từ đầu bằng NumPy
├── problem2_multiclass_and_thresholds.ipynb  # Problem 2: So sánh OvR vs Multinomial & Tối ưu hóa Ngưỡng quyết định
└── scripts/
    ├── build_lr_notebooks.py                 # Script tự động tạo và thực thi 2 notebooks kèm đầy đủ outputs
    └── test_notebooks_execution.py           # Script kiểm thử thực thi toàn vẹn độc lập
```

---

## 2. Tóm tắt Bộ Dữ liệu & Thiết lập Thực nghiệm

* **Nguồn dữ liệu:** [Seeds Dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds), DOI [10.24432/C5538F](https://doi.org/10.24432/C5538F), License CC BY 4.0.
* **Quy mô:** 210 mẫu hạt giống, 7 đặc trưng hình thái học dạng số liên tục (`Area`, `Perimeter`, `Compactness`, `Kernel_Length`, `Kernel_Width`, `Asymmetry_Coeff`, `Kernel_Groove`).
* **Nhãn mục tiêu (3 lớp cân bằng hoàn hảo):**
  * `0: Kama` (70 mẫu)
  * `1: Rosa` (70 mẫu)
  * `2: Canadian` (70 mẫu)
* **Phân hoạch dữ liệu (`SEED=42`):**
  * Chia phân tầng (Stratified): **60% Train (126 mẫu)**, **20% Validation (42 mẫu)**, **20% Test (42 mẫu)**.
  * Đảm bảo tính độc lập tuyệt đối giữa các tập (`assert len(set(train) & set(val)) == 0`).
  * Chuẩn hóa đặc trưng: `StandardScaler().fit(X_train)`, sau đó transform độc lập cho Validation và Test để chống rò rỉ dữ liệu (No Data Leakage).

---

## 3. Tổng hợp Kết quả Thực nghiệm Chính

### Problem 1 — Logistic Regression từ First Principles (NumPy)
* **Thuật toán tự xây dựng:** Sigmoid ổn định số học, hàm mất mát Log Loss (Binary Cross-Entropy), đạo hàm giải tích Gradient Descent, mở rộng One-vs-Rest (OvR) với quy tắc $\arg\max$.
* **Động lực học hội tụ:**
  * Cả 3 mô hình nhị phân con khởi đầu từ $J_0 \approx 0.6931$ ($-\ln 0.5$) và hội tụ ổn định sau 1500 epoch (gradient norm suy giảm tiệm cận 0).
  * Đường cong Validation Loss bám sát Training Loss, không xảy ra quá khớp (overfitting).
* **Đối chứng với scikit-learn:**
  * **NumPy OvR (Tự xây dựng):** Validation Accuracy = **0.9048**, Macro-F1 = **0.9027**, Log Loss = **0.2075**.
  * **scikit-learn OneVsRestClassifier:** Validation Accuracy = **0.8810**, Macro-F1 = **0.8792**, Log Loss = **0.2219**.
  * Bản cài đặt từ First Principles đạt hiệu năng ngang ngửa, thậm chí nhỉnh hơn mô hình thư viện nhờ không bị ràng buộc bởi hệ số phạt L2 mặc định.

---

### Problem 2 — So sánh Mô hình Đa lớp & Phân tích Ngưỡng Quyết định

#### Phần A: So sánh Đa lớp (OvR vs Multinomial) trên Validation
| Mô hình | Validation Accuracy | Validation Macro F1 | Validation Log Loss |
|---|:---:|:---:|:---:|
| **One-vs-Rest (OvR)** | 0.8810 | 0.8792 | 0.2219 |
| **Multinomial (Softmax)** | **0.9048** | **0.9027** | **0.1666** |

> [!NOTE]
> **Nhận xét then chốt:** Mô hình Multinomial vượt trội hơn hẳn OvR về độ mất mát Log Loss (giảm $25\%$ sai số cross-entropy). Cơ chế Softmax tối ưu hóa đồng thời và tạo ra sự cạnh tranh trực tiếp giữa các lớp trong gradient, giúp phân phối xác suất đầu ra được hiệu chuẩn (calibrated) chuẩn xác hơn so với phép chuẩn hóa sau heuristic của OvR.

#### Phần B: Phân tích Ngưỡng Quyết định & Hàm Chi phí Lỗi
* **Thiết lập:** Lớp *Canadian* ($y=2$) là Positive, *Kama & Rosa* là Negative.
* **Hàm chi phí lỗi:** $\text{Cost} = 1 \cdot FP + 5 \cdot FN$ (phạt nặng bỏ sót hạt giống thuần chủng thượng hạng).
* **Khảo sát ngưỡng trên Validation:**
  * Tại ngưỡng mặc định $\tau = 0.50$: Precision = $0.8667$, Recall = $0.9286$, $FP=2, FN=1 \implies \mathbf{\text{Cost} = 7.0}$.
  * Tại ngưỡng tối ưu $\tau^* = 0.25$: Precision = $0.8750$, **Recall = 1.0000 (100%)**, $FP=2, FN=0 \implies \mathbf{\text{Cost} = 2.0}$ (giảm hơn $71\%$ chi phí).
* **Đánh giá trên tập Test nguyên bản (Đóng băng mô hình & ngưỡng):**
  * Mô hình đạt **Test Accuracy = 0.8810**, **Macro-F1 = 0.8731**.
  * Tại ngưỡng $\tau^*$, mô hình đạt **Recall tuyệt đối 100%** trên toàn bộ 14 mẫu Canadian thực tế trong tập Test ($FN=0$).
* **Đối chứng Baseline & Accuracy Paradox:**
  * Mô hình tầm thường Dummy Baseline (luôn đoán lớp âm tính) đạt Accuracy $= 66.67\%$, nhưng Recall $= 0\%$ và chi phí lỗi vọt lên tới $\mathbf{70.0}$.
  * Mô hình Multinomial của chúng ta cắt giảm tới **$94.3\%$ chi phí thiệt hại** so với baseline, chứng minh rằng trong bài toán mất cân bằng hoặc có chi phí bất đối xứng, Accuracy là chỉ số dễ gây ngộ nhận nguy hiểm.

---

## 4. Hướng dẫn Tái lập Thực nghiệm (Reproducibility)

Tất cả các tệp notebook đã được tạo sẵn và lưu kèm toàn bộ kết quả chạy (bảng số liệu, biểu đồ trực quan hóa). Để chạy lại toàn bộ hoặc kiểm thử:

### Cách 1: Chạy lại toàn bộ bằng Script
Từ thư mục gốc repository hoặc từ `logistic-regression/`:
```powershell
python logistic-regression/scripts/build_lr_notebooks.py
```

### Cách 2: Kiểm thử tính toàn vẹn độc lập
```powershell
python logistic-regression/scripts/test_notebooks_execution.py
```

### Cách 3: Chạy trực tiếp trên JupyterLab / VS Code / Google Colab
Mở `problem1_first_principles_lr.ipynb` hoặc `problem2_multiclass_and_thresholds.ipynb` và chọn **Kernel → Restart & Run All**. Notebook hỗ trợ cơ chế tự động tải và kiểm tra mã băm SHA-256 từ UCI nếu chạy trên Colab.
