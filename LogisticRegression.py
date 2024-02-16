import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_score, accuracy_score, recall_score, f1_score
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

#---------------TIỀN XỬ LÝ DỮ LIỆU---------------
# Hàm để xử lý cột Blood Pressure
def systolic_column(blood_pressure):
    systolic = int(re.findall('\w{2,}\/', blood_pressure)[0][:-1])
    return systolic

def diastolic_column(blood_pressure):
    diastolic = int(re.findall('\/\w{2,}', blood_pressure)[0][1:])
    return diastolic

# Đọc dữ liệu
df = pd.read_csv('./Data/heart_attack_prediction_dataset.csv')

# Drop missing value
df = df.dropna()

# Tách cột Blood Pressure thành 2 cột Systolic và Diastolic
df['Systolic']= df['Blood Pressure'].apply(systolic_column)
df['Diastolic'] = df['Blood Pressure'].apply(diastolic_column)

col_new = ['Systolic', 'Diastolic']
df[col_new].to_csv('./Data/blood_pressure.csv', index=False)

# Xóa các cột không cần thiết
delete_column = ['Patient ID', 'Blood Pressure', 'Continent', 'Hemisphere']
df = df.drop(columns=delete_column)

# Chia feature và target
target = 'Heart Attack Risk'
x = df.drop(target, axis=1)
y = df[target]

# Chia test và train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Tiền xử lý dữ liệu
num_column = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week',
              'Stress Level', 'Sedentary Hours Per Day','Income', 'BMI', 'Triglycerides',
              'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Systolic', 'Diastolic']

diet_values = ['Healthy', 'Average', 'Unhealthy']
sex_values = df['Sex'].unique()

processor = ColumnTransformer([
    ('num_feature', StandardScaler(), num_column),
    ('ord_feature', OrdinalEncoder(categories=[sex_values, diet_values]), ['Sex', 'Diet']),
    ('nom_feature', OneHotEncoder(), ['Country'])
])

x_train_transformed = processor.fit_transform(x_train)
x_test_transformed = processor.fit_transform(x_test)
df.to_csv('./Data/data_after_preprocessing.csv', index=False)

#---------------PHÂN TÍCH MÔ TẢ---------------

df1 = pd.read_csv('Data/heart_attack_prediction_dataset.csv')
df2 = pd.read_csv('Data/data_after_preprocessing.csv')
df_numeric = df2.select_dtypes(include=['number'])

# Tạo biểu đồ cột với các cột phi số
ctg_data = df1[['Sex', 'Diet', 'Continent', 'Country', 'Hemisphere']]

plt.figure(figsize=(12, 8))
for i in ctg_data.columns:
    ctg_num = ctg_data[i].value_counts()
    chart = sns.barplot(x=ctg_num.index, y=ctg_num)
    for p in chart.patches:
        chart.annotate(format(p.get_height(), '.0f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 10),
                       textcoords='offset points',
                       fontsize=8)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Tạo biểu đồ Boxplot với các cột số và phi số
plt.figure(figsize=(9, 4))
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(x="Continent", y='Cholesterol', hue="Sex",
            palette=["m", "g"],
            data=df1)
sns.despine(offset=10, trim=True)

plt.figure(figsize=(9, 4))
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(x="Continent", y='Exercise Hours Per Week', hue="Sex",
            palette=["b", "r"],
            data=df1)
sns.despine(offset=10, trim=True)

plt.figure(figsize=(9, 4))
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(x="Continent", y='Stress Level', hue="Sex",
            palette=["y", "r"],
            data=df1)
sns.despine(offset=10, trim=True)

# Tạo biểu đồ nhiệt
df_numeric.corr(numeric_only=True)
## Tạo biểu đồ thể hiện mức độ tương quan giữa các thuộc tính
plt.figure(figsize=(12, 10))
sns.heatmap(df_numeric.corr(numeric_only=True), annot=True, cmap=plt.cm.Blues)
plt.xticks(rotation=45, ha='right')
plt.show()

# Tạo bảng thống kê
print("-------------Bảng thông kê mô tả với dữ liệu kiểu số------------------------------")
data_complete = df_numeric.describe(include='all')
print(data_complete)
data_complete.to_csv('Thong_ke_2.txt', sep='\t', index=False)

#---------------PHÂN TÍCH HỒI QUY LOGISTIC---------------

# Huấn luyện mô hình
logistic_model = LogisticRegression()
logistic_model.fit(x_train_transformed, y_train)

# Lưu mô hình
def save_model(filename, model):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

save_model('logistic_model.pkl', logistic_model)

# Sử dụng mô hình dự đoán trên tập test
y_pred = logistic_model.predict(x_test_transformed)

# Kết quả test mô hình
def print_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, zero_division=1)

    print("Classification report:")
    print(report)

print('Kết quả thực hiện dự đoán bằng mô hình Logistic:')
print_classification_report(y_test, y_pred)

# Biểu đồ Confusion matrix
def show_confusion_matrix(model, x_test, y_test):
    titles_options = [
        ("Confusion matrix không chuẩn hoá", None),
        ("Confusion matrix chuẩn hoá", "true")
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            x_test,
            y_test,
            cmap="Blues",
            normalize=normalize,
        )
        disp.ax_.set_title(title)

    plt.show()

show_confusion_matrix(logistic_model, x_test_transformed, y_test)

# Cải thiện mô hình Logistic

# Điều chinh trọng số của lớp cho mô hình
logistic_model_improved = LogisticRegression(class_weight='balanced')
logistic_model_improved.fit(x_train_transformed, y_train)

# Lưu mô hình sau khi điều chỉnh trọng số của lớp
save_model('logistic_model_improved.pkl', logistic_model_improved)

# Dự đoán bằng mô hình sau khi điều chỉnh trọng số của lớp
y_pred = logistic_model_improved.predict(x_test_transformed)
print('Kết quả thực hiện dự đoán bằng mô hình Logistic sau khi điều chỉnh trọng số của lớp:')
print_classification_report(y_test, y_pred)

# Tiếp tục vẽ biểu đồ cho confusion matrix
show_confusion_matrix(logistic_model_improved, x_test_transformed, y_test)

#---------------ĐÁNH GIÁ VÀ SO SÁNH CÁC MÔ HÌNH---------------

# Hàm train và đánh giá mô hình
def train_and_evaluate(model, model_name, X_train, X_test, Y_train, Y_test):
    start_time = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    Y_pred = model.predict(X_test)
    test_time = time.time() - start_time

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=1)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    return {
      "Model": model_name,
      "Train Time (s)": round(train_time, 2),
      "Test Time (s)": round(test_time, 2),
      "Accuracy": round(accuracy, 2),
      "Precision": round(precision, 2),
      "Recall": round(recall, 2),
      "F1-Score": round(f1, 2),
    }

# Thực hiện train và đánh giá mô hình
models = [
  LogisticRegression(class_weight='balanced'),
  KNeighborsClassifier(n_neighbors=10),
  GaussianNB(),
  SVC(),
  RandomForestClassifier()
]

results = []
for model in models:
    model_name = model.__class__.__name__
    if(model_name == 'LogisticRegression'):
        model_name = "Logistic Regression"
    elif(model_name == 'KNeighborsClassifier'):
        model_name = "KNN"
    elif(model_name == 'GaussianNB'):
        model_name = "Naive Bayes"
    elif(model_name == 'SVC'):
        model_name = "SVM"
    elif(model_name == 'RandomForestClassifier'):
        model_name = "Random Forest"
    results.append(train_and_evaluate(model, model_name, x_train_transformed, x_test_transformed, y_train, y_test))

# In ra màn hình
print("Bảng so sánh các mô hình:")
df_result = pd.DataFrame(results, index=None)
print(df_result.to_string())