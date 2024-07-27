# Import các thư viện cần thiết
import numpy as np 
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

print("Libraries imported")  # In ra thông báo các thư viện đã được nhập

# Hiển thị dataset
df = pd.read_csv("D:\\Study stuff\\Machine Learning\\datasets\\apple_quality.csv")

df.head()  # Hiển thị 5 dòng đầu tiên của dataset

df.info()  # Thông tin tổng quan về dataset, bao gồm số lượng các giá trị null, kiểu dữ liệu

df = df.drop('A_id', axis=1)  # Loại bỏ cột 'A_id' không cần thiết

print("Number of duplicates", df.duplicated().sum())  # Đếm số lượng các dòng trùng lặp
print("Number of null values:")  # Đếm số lượng các giá trị null cho từng cột
df.isna().sum()

df[df.isnull().any(axis=1)]  # Hiển thị các dòng có giá trị null
df['Acidity'] = df['Acidity'].astype(float)  # Chuyển đổi kiểu dữ liệu của cột 'Acidity' sang float
df['Quality'] = df['Quality'].map({'good': 1, 'bad': 0})  # Mapping cột 'Quality' thành giá trị nhị phân

df.head()  # Hiển thị 5 dòng đầu tiên sau khi xử lý

# Vẽ biểu đồ ma trận tương quan
fig = ff.create_annotated_heatmap(z=df.corr().values, x=list(df.corr().columns), y=list(df.corr().index),
                                  colorscale='Blues', annotation_text=df.corr().round(2).values, showscale=True)

fig.update_layout(title="Correlation Heatmap", width=850, height=850)

fig.show()  # Hiển thị biểu đồ

# Các cột dữ liệu cần hiển thị
columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

# Tạo biểu đồ scatter plot matrix với các tùy chỉnh
fig = px.scatter_matrix(df, dimensions=columns, color='Quality', color_continuous_scale='Portland_r', 
                        range_color=[0, 1.62], title='Scatter Plot Matrix of DataFrame Columns', opacity=0.5)

fig.update_layout(coloraxis_showscale=False)

# Cập nhật layout để dễ đọc hơn
fig.update_layout(width=1100, height=1100)

# Hiển thị biểu đồ
fig.show()

# Chuyển đổi DataFrame để có các hàng riêng biệt cho từng mức chất lượng
melted_df = pd.melt(df, id_vars=['Quality'], var_name='Column')

# Biểu đồ box plot cho từng cột
fig = px.box(melted_df, x='Column', y='value', color='Quality',
             title='Box Plots for Each Column Grouped by Quality',
             color_discrete_map={1: 'green', 0: 'red'})

# Cập nhật nhãn
fig.for_each_trace(lambda t: t.update(name="Good" if t.name == "1" else "Bad"))

fig.show()  # Hiển thị biểu đồ

# Tạo các dataset riêng biệt cho các quả táo chất lượng tốt và xấu
good_quality = df[df['Quality'] == 1]
bad_quality = df[df['Quality'] == 0]

bin_number = 100  # Số lượng bins trong biểu đồ histogram

# Tạo các biểu đồ histogram riêng biệt cho từng cột
for column in df.columns:
    if column != 'Quality':
        fig = go.Figure()

        fig.add_trace(go.Histogram(x=good_quality[column], name='Good Quality', opacity=0.6, marker_color='green', nbinsx=bin_number))
        fig.add_trace(go.Histogram(x=bad_quality[column], name='Bad Quality', opacity=0.5, marker_color='red', nbinsx=bin_number))
        
        fig.update_layout(title=f'Histogram of {column} Grouped by Quality', xaxis_title='Value', yaxis_title='Frequency', bargap=.1)

        fig.show()

from plotly.subplots import make_subplots

# Tạo subplot với lưới các biểu đồ histogram
fig = make_subplots(rows=4, cols=2, subplot_titles=columns, vertical_spacing=0.1, horizontal_spacing=0.1)

# Thêm histogram cho từng cột
for i, column in enumerate(columns):
    row = i // 2 + 1
    col = i % 2 + 1
    fig.add_trace(go.Histogram(x=df[column], name=column, opacity=0.7), row=row, col=col)
    fig.update_xaxes(title_text='Value')
    fig.update_yaxes(title_text='Frequency')

fig.update_layout(title_text='Histograms of DataFrame Columns', showlegend=False, width=1000, height=1200)

fig.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Định nghĩa features và target
X = df.drop(["Quality"], axis=1)
y = df["Quality"]

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Tạo và huấn luyện mô hình Logistic Regression
lr_model = LogisticRegression()  
lr_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = lr_model.predict(X_test)

# In kết quả đánh giá mô hình
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier

# Khởi tạo mô hình k-NN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = knn_model.predict(X_test)

# In kết quả đánh giá mô hình
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

from sklearn.svm import SVC

# Khởi tạo mô hình SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = svm_model.predict(X_test)

# In kết quả đánh giá mô hình
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Xây dựng mô hình mạng neural
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

# Dự đoán trên tập test
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# In kết quả đánh giá mô hình
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

import tensorflow as tf

# Giả định 'model' là mô hình đã được huấn luyện
# Lưu mô hình vào đường dẫn cụ thể
model.save('D:\\Study stuff\\Machine Learning\\Git_Demo_CDT_01_K27\\ML_CDT\\apple quality\\apple quality.keras')

# Hiển thị trên Streamlit
import streamlit as st

st.title('Apple Quality')  # Tiêu đề ứng dụng
st.header('by =')  # Tên tác giả hoặc nhóm
st.markdown('[My GitHub](https://github.com/EqualVu/ML_CDT/tree/baihoctuan/t4)')  # Đường dẫn tới GitHub

st.divider()  # Thêm đường ngăn cách
files = st.file_uploader('Please upload file: ', accept_multiple_files=True)  # Widget tải lên file
for file in files:
    read_f = file.read()
    st.write('File name: ', file.name)
st.divider()  # Thêm đường ngăn cách

import nbformat

# Đường dẫn tới notebook Jupyter
notebook_path = 'D:\\Study stuff\\Machine Learning\\Git_Demo_CDT_01_K27\\ML_CDT\\apple quality\\Apple Quality.ipynb'

# Đường dẫn lưu mã Python đã trích xuất
output_path = 'D:\\Study stuff\\Machine Learning\\Git_Demo_CDT_01_K27\\ML_CDT\\apple quality\\Apple Quality.py'

# Tải notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Trích xuất các cell mã
code_cells = [cell['source'] for cell in nb.cells if cell.cell_type == 'code']

# Ghi vào file script Python


# Chuyển code sang python file
with open(output_path, 'w', encoding='utf-8') as f:
    for cell in code_cells:
        f.write(cell + '\n\n')

print(f'Code extracted and saved to {output_path}')


