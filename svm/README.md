# Thực hành SVM — Rice (Cammeo và Osmancik)

Bài thực hành dùng 3.810 quan sát hạt gạo, 7 đặc trưng hình thái dạng số và hai nhãn `Cammeo`/`Osmancik` để khảo sát SVM tuyến tính, polynomial và RBF. Đề đầy đủ nằm trong [SVM.md](SVM.md); dữ liệu lấy từ [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice%2Bcammeo%2Band%2Bosmancik), DOI [10.24432/C5MW4Z](https://doi.org/10.24432/C5MW4Z), giấy phép CC BY 4.0.

## Cấu trúc bài

```text
svm/
├── README.md                         # Hướng dẫn này
├── SVM.md                            # Đề bài gốc
├── problem1_linear_svm.ipynb         # Linear SVM, scaling, C và minh họa margin
├── problem2_kernel_svm.ipynb         # Polynomial/RBF, tuning và đánh giá test
├── data/Rice_Cammeo_Osmancik.arff    # Dữ liệu gốc từ UCI
└── scripts/build_svm_notebooks.py    # Tạo lại hai notebook từ mã nguồn
```

Hai notebook tự chứa phần tải, chia dữ liệu và chuẩn hóa; có thể chạy độc lập. Tệp trong `scripts/` chỉ cần dùng khi muốn tạo lại notebook, không cần chạy để làm bài.

## Chuẩn bị môi trường

Từ thư mục gốc repository, tạo và kích hoạt môi trường Conda đã khai báo sẵn:

```powershell
conda env create -f environment.yml
conda activate ml-implementation
cd svm
jupyter lab
```

Nếu môi trường `ml-implementation` đã tồn tại, chỉ cần kích hoạt và mở JupyterLab. Bài dùng NumPy, pandas, SciPy, scikit-learn và Matplotlib từ `environment.yml`; không cần cài thêm gói.

Mở lần lượt `problem1_linear_svm.ipynb` và `problem2_kernel_svm.ipynb`, chọn kernel của môi trường trên, rồi **Run All** từ kernel sạch. Có thể chạy notebook từ thư mục `svm/` hoặc từ gốc repository. Notebook ưu tiên tệp ARFF trong cây thư mục và tự tải từ UCI nếu không tìm thấy.

### Chạy trên Google Colab

Tải từng notebook lên Colab và chọn **Runtime → Run all**. Ô chuẩn bị dữ liệu dùng tệp ARFF có sẵn nếu tìm thấy; nếu không, ô này tự tải archive Rice chính thức từ UCI và kiểm tra SHA-256 trước khi lưu vào `data/`. Lần tải đầu cần kết nối Internet. Tệp trong runtime Colab chỉ tồn tại trong phiên làm việc. Hai notebook chạy độc lập và không cần upload dataset thủ công. Các thư viện NumPy, pandas, SciPy, scikit-learn và Matplotlib cần có sẵn trong runtime.

## Trình tự thực hành

1. **Kiểm tra dữ liệu:** xác nhận 3.810 mẫu, 7 đặc trưng, hai lớp, kiểu dữ liệu, missing values, duplicate rows và phân bố nhãn. Notebook báo cáo duplicate nhưng không tự ý xóa.
2. **Chia dữ liệu:** dùng `random_state=42`, stratified train/validation/test theo tỷ lệ 60/20/20 (2.286/762/762 mẫu). Fit `StandardScaler` trên train; dùng cùng split cho mọi mô hình.
3. **Problem 1 — tuyến tính:** so sánh có/không scaling tại `C=1`; thử `C = 0.01, 0.1, 1, 10, 100` trên đủ 7 đặc trưng. Đọc bảng train/validation macro-F1, accuracy, balanced accuracy, margin width và số support vectors. Với hai đặc trưng `Major_Axis_Length`/`Eccentricity`, xem boundary, hai đường margin và đối chiếu phép tính NumPy với `decision_function`.
4. **Problem 2 — kernel:** thử linear, polynomial (`degree = 2, 3, 4`) và RBF (`gamma = 0.001, 0.01, 0.1, 1, 10`) trên cùng grid `C`. Dùng validation macro-F1 để chọn cấu hình; các heatmap giúp đọc xu hướng underfitting/overfitting. So sánh biên linear/RBF trên cùng dữ liệu hai chiều.
5. **Đánh giá cuối:** sau khi chốt tham số, đọc bảng test và confusion matrix của ba kernel. Khuyến nghị mô hình dựa trên validation; test chỉ dùng để ước lượng khả năng tổng quát hóa.

Macro-F1 là metric chính vì cho hai lớp trọng số bằng nhau và không phụ thuộc lớp nào được chọn làm positive class. Accuracy và balanced accuracy là thông tin bổ sung. Nếu các cấu hình cùng kernel chênh dưới `1e-4` validation macro-F1, notebook chọn `C` nhỏ hơn, rồi `degree`/`gamma` nhỏ hơn. Khi chọn kernel để khuyến nghị, nếu chênh dưới `0.001`, ưu tiên linear, rồi polynomial, rồi RBF.

## Kết quả cần kiểm tra và nộp

- Nộp **hai notebook**; mỗi notebook phải chạy lại được từ kernel sạch, có code, bảng, biểu đồ và giải thích tiếng Việt.
- Problem 1 phải có bảng ảnh hưởng của `C`, biểu đồ hiệu năng và support vectors, hình boundary/margin, bảng decision score/signed distance/hinge loss và giải thích hard/soft margin.
- Problem 2 phải có bảng tuning, heatmap polynomial/RBF, hình so sánh linear/RBF, bảng test, confusion matrix, khuyến nghị mô hình và giới hạn của kết luận.
- Trước khi nộp, kiểm tra các ô kết quả đã được chạy theo thứ tự, không còn lỗi, nguồn dữ liệu và seed vẫn được ghi rõ. Nếu hệ thống nộp bài không tự cung cấp dữ liệu, gửi kèm `data/Rice_Cammeo_Osmancik.arff` để notebook chạy độc lập.

Nếu môi trường có `nbconvert`, có thể kiểm tra bằng dòng lệnh từ thư mục `svm/`. Nếu không, dùng **Run All** trong JupyterLab:

```powershell
jupyter nbconvert --to notebook --execute problem1_linear_svm.ipynb --output problem1_checked.ipynb
jupyter nbconvert --to notebook --execute problem2_kernel_svm.ipynb --output problem2_checked.ipynb
```

Các tệp `*_checked.ipynb` chỉ là bản kiểm tra; hai notebook gốc mới là sản phẩm chính.
