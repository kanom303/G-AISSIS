import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 👇 เพิ่มไลบรารีสำหรับตัดคำภาษาไทยตรงนี้ (ถ้ายังไม่มีให้พิมพ์ pip install pythainlp ใน Terminal ก่อนนะ)
from pythainlp.tokenize import word_tokenize

# ==============================
# 1️⃣ โหลดข้อมูล
# ==============================

data = pd.read_csv("dataset.csv")

# ลบแถวที่ว่าง
data = data.dropna()

# แปลงข้อความเป็น string
data["text"] = data["text"].astype(str)

print("จำนวนข้อมูลทั้งหมด:", len(data))


# ==============================
# 2️⃣ แยก X และ y
# ==============================

X = data["text"]
y = data["label"]


# ==============================
# 3️⃣ Stratified Split (สำคัญมาก)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 👈 ทำให้แต่ละคลาสกระจายเท่าๆกัน
)


# ==============================
# 4️⃣ TF-IDF + NGRAM (1,2)
# ==============================

# 👇 เพิ่มฟังก์ชันสำหรับตัดคำภาษาไทย
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

# 👇 ปรับ Vectorizer โดยเรียกใช้ฟังก์ชันตัดคำภาษาไทย
vectorizer = TfidfVectorizer(
    tokenizer=thai_tokenizer,  # 👈 ใส่ตัวตัดคำภาษาไทยเข้าไปตรงนี้
    ngram_range=(1, 2),        # 👈 สำคัญมาก
    max_features=5000,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ==============================
# 5️⃣ โมเดล (Logistic Regression)
# ==============================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"   # 👈 ช่วยแก้ class ไม่เท่ากัน
)

model.fit(X_train_vec, y_train)


# ==============================
# 6️⃣ ประเมินผล
# ==============================

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nรายละเอียดแต่ละหมวด:")
print(classification_report(y_test, y_pred, zero_division=0))


# ==============================
# 7️⃣ บันทึกโมเดล
# ==============================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nบันทึก model.pkl และ vectorizer.pkl เรียบร้อยแล้ว ✅")