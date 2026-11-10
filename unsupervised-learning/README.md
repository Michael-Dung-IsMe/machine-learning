# Thực hành học không giám sát: K-means và phân cụm phân cấp

## 1. Mục tiêu

Bài thực hành huấn luyện và so sánh **K-means** với **Agglomerative Clustering** trên cùng một bộ quan sát và cùng tập đặc trưng số. Nội dung phải làm rõ ảnh hưởng của chuẩn hóa dữ liệu, số cụm, cách khởi tạo và linkage đến cấu trúc cụm thu được.

Đề bài đầy đủ nằm tại [Unsupervised-Learning.md](Unsupervised-Learning.md). Xem [kế hoạch](docs/implementation-plan.md), [kiểm tra nghiệm thu](docs/acceptance-tests.md) và [báo cáo kết quả](docs/report.md).

## 2. Dữ liệu và thiết kế chung đã chọn

Sử dụng **Rice Cammeo and Osmancik** từ [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik): 3.810 quan sát và 7 đặc trưng hình thái dạng số. Dataset đáp ứng yêu cầu tối thiểu 1.000 mẫu và 4 đặc trưng số, đồng thời có kích thước đủ để khảo sát chi phí của phân cụm phân cấp.

- Dùng cả 7 đặc trưng số cho thí nghiệm chính; cột nhãn giống lúa không được đưa vào `X` hoặc dùng để chọn mô hình.
- K-means và phân cụm phân cấp phải dùng cùng hàng dữ liệu, cùng đặc trưng và cùng phiên bản dữ liệu đã tiền xử lý trong từng phép so sánh.
- So sánh dữ liệu gốc với dữ liệu qua `StandardScaler` để đo ảnh hưởng của scale.
- Dùng `SEED = 42` cho lấy mẫu, khởi tạo và mọi thao tác ngẫu nhiên có thể kiểm soát.
- Nếu dùng nhãn thật để tính ARI/NMI, chỉ thực hiện như đánh giá hậu kiểm sau khi đã chốt cấu hình bằng tiêu chí không giám sát.

> Đây là bài toán khám phá không giám sát nên không áp dụng máy móc train/validation/test như bài phân loại. Toàn bộ quy tắc chọn cấu hình phải được công bố trước; nhãn thật không tham gia preprocessing, huấn luyện hoặc lựa chọn số cụm.

## 3. Các đầu việc đã triển khai

### 3.1. Chuẩn bị và khảo sát dữ liệu

- Ghi nguồn, giấy phép, mô tả 7 đặc trưng và lý do lựa chọn Rice.
- Kiểm tra shape, kiểu dữ liệu, missing values, giá trị vô hạn, hàng trùng và thống kê mô tả.
- Phân tích thang đo, phân phối, tương quan và dấu hiệu outlier; nêu rõ cách xử lý nhưng không tự ý xóa dữ liệu.
- Tạo hai biểu diễn nhất quán: dữ liệu gốc và dữ liệu chuẩn hóa.

### 3.2. Problem 1 — K-means (40 điểm)

- Huấn luyện `KMeans` bằng scikit-learn với `K = 2..10`.
- Đánh giá bằng inertia, silhouette score, kích thước cụm, thời gian chạy và trực quan hóa cụm.
- So sánh `init="random"` với `init="k-means++"` qua nhiều seed/lần lặp; báo cáo độ ổn định của inertia, silhouette và phân hoạch.
- Chọn hai đặc trưng số và một tập quan sát thật để tự thực hiện đúng **một bước gán cụm** và **một bước cập nhật centroid** bằng NumPy.
- Trình bày nhãn gán, centroid trước/sau, inertia trước/sau; giải thích vì sao mỗi bước không làm tăng hàm mục tiêu và vì sao hội tụ chỉ bảo đảm nghiệm tối ưu cục bộ.

### 3.3. Problem 2 — Hierarchical Clustering (60 điểm)

- Huấn luyện `AgglomerativeClustering` với `single`, `complete`, `average` và `ward`.
- Dùng Euclidean cho phép so sánh chính; ghi rõ Ward chỉ tương thích với Euclidean.
- Khảo sát cùng dải số cụm với K-means và báo cáo silhouette, Calinski–Harabasz, Davies–Bouldin, kích thước cụm và runtime.
- Vẽ dendrogram cho ít nhất hai linkage, ưu tiên Ward và average; giải thích merge height và vị trí cắt cây.
- Phân tích ảnh hưởng của linkage, số cụm, outlier, hình dạng cụm và chi phí tính toán.
- Nếu dendrogram toàn bộ dữ liệu quá nặng, dùng mẫu ngẫu nhiên tái lập và giải thích kích thước/cách lấy mẫu; phép so sánh mô hình chính vẫn dùng cùng tập quan sát khi tài nguyên cho phép.

### 3.4. So sánh, diễn giải và khuyến nghị

- Lập một bảng chung cho các cấu hình, gồm preprocessing, thuật toán/linkage, số cụm, metric nội tại, kích thước cụm và runtime.
- Trực quan hóa trên hai đặc trưng gốc đã chọn và/hoặc PCA 2D; ghi rõ hình 2D chỉ mang tính minh họa.
- Diễn giải từng cụm bằng thống kê các đặc trưng gốc, không chỉ bằng tọa độ đã scale.
- Chọn phương pháp khuyến nghị theo quy tắc đã định, đánh giá tính hữu ích và nêu hạn chế của kết luận.

### 3.5. Sản phẩm nộp

- `problem1_kmeans.ipynb`: toàn bộ Problem 1, gồm thí nghiệm scaling/K/init và bước NumPy minh họa.
- `problem2_hierarchical.ipynb`: toàn bộ Problem 2, dendrogram, bảng so sánh chung và khuyến nghị.
- Hai notebook phải chạy độc lập từ kernel sạch, chứa code, bảng, biểu đồ và giải thích tiếng Việt; code, tên biến và docstring dùng tiếng Anh.
- Phần mở rộng imputation/outlier detection là tùy chọn, không được làm ảnh hưởng các yêu cầu bắt buộc.

## 4. Cấu trúc thư mục

```text
unsupervised-learning/
├── README.md
├── Unsupervised-Learning.md
├── problem1_kmeans.ipynb
├── problem2_hierarchical.ipynb
└── data/
    └── Rice_Cammeo_Osmancik.arff
```

Hai notebook đã được tạo từ `scripts/build_notebooks.py`, chạy bằng môi trường `ai-algorithms` và lưu output. File ARFF có checksum SHA-256 được kiểm tra khi chạy notebook.

## 5. Môi trường và cách kiểm tra

Thực nghiệm đã chạy trong môi trường `ai-algorithms` hiện có. Từ thư mục gốc repository:

```powershell
conda activate ai-algorithms
jupyter lab
```

Chạy từng notebook từ kernel sạch. Lệnh đã dùng để kiểm tra:

```powershell
jupyter nbconvert --to notebook --execute unsupervised-learning/problem1_kmeans.ipynb --output problem1_kmeans.ipynb --output-dir unsupervised-learning
jupyter nbconvert --to notebook --execute unsupervised-learning/problem2_hierarchical.ipynb --output problem2_hierarchical.ipynb --output-dir unsupervised-learning
```

Mỗi notebook có thể chạy từ thư mục gốc hoặc từ `unsupervised-learning/`. Khi muốn sửa nội dung notebook, sửa `scripts/build_notebooks.py`, tạo lại hai notebook, rồi chạy lại để output và nhận xét khớp nhau.
