import streamlit as st
import pickle
from pythainlp.tokenize import word_tokenize

# 1. ฟังก์ชันตัดคำ (ต้องมีเพื่อให้โหลด Vectorizer ได้)
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

# 2. โหลดโมเดล (ใช้ cache เพื่อให้หน้าเว็บทำงานไวขึ้น)
@st.cache_resource
def load_system():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_system()

# 3. จัดหน้าตาเว็บแอป
st.title("🤖 G-Assist ")
st.write("ลองพิมพ์ปัญหาที่พบด้านล่าง ระบบจะจัดหมวดหมู่แจ้งปัญหา (Ticket) ให้โดยอัตโนมัติ")

# กล่องรับข้อความ
user_input = st.text_area("💬 ข้อความจากผู้เล่น:", placeholder="ตัวอย่าง: เติมเงินเข้าเกมผ่านพร้อมเพย์แล้วเพชรไม่ขึ้นครับ...")

# ปุ่มกด
if st.button("🔍 วิเคราะห์ปัญหา"):
    if user_input.strip() == "":
        st.warning("⚠️ กรุณาพิมพ์ข้อความก่อนกดวิเคราะห์ครับ")
    else:
        # กระบวนการให้ AI ทำนาย
        test_vec = vectorizer.transform([user_input])
        prediction = model.predict(test_vec)[0]
        
        # แสดงผลลัพธ์
        st.success(f"🏷️ AI จัดหมวดหมู่นี้ให้อยู่ในกลุ่ม: **{prediction}**")
