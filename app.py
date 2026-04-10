import streamlit as st
import pickle
from pythainlp.tokenize import word_tokenize
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="G-Assist | AI Ticket Classifier",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS - Dark Gaming Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

body {
    font-family: 'Inter', sans-serif !important;
    background-color: #0d0f1a !important;
    color: #e2e8f0 !important;
}

.stApp {
    background-color: #0d0f1a !important;
}

#MainMenu { visibility: hidden; }
.viewerBadge_container__1QSob { display: none; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f1a; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
}

/* Text Area */
.stTextArea textarea {
    background: #0d0f1a !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    min-height: 140px !important;
}

.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}

.stTextArea textarea::placeholder { color: #334155 !important; }

/* Select Box */
.stSelectbox select {
    background: #0d0f1a !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 0.5rem !important;
}

/* Markdown Styling */
h1, h2, h3 {
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}

p {
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Table Styling */
table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
}

table th {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    padding: 0.75rem !important;
    text-align: left !important;
    border-bottom: 1px solid rgba(99,102,241,0.3) !important;
    font-weight: 700 !important;
}

table td {
    padding: 0.75rem !important;
    border-bottom: 1px solid rgba(99,102,241,0.1) !important;
    color: #cbd5e1 !important;
}

table tr:hover td {
    background: rgba(99,102,241,0.05) !important;
}

/* Alert Styling */
.stAlert {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

@st.cache_resource
def load_system():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_system()

# ─────────────────────────────────────────────────────────────────────────────
# Session State - REAL-TIME ONLY (No Mock Data)
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "analyze"

if "history" not in st.session_state:
    st.session_state.history = []  # Empty - only real data

if "ticket_counter" not in st.session_state:
    st.session_state.ticket_counter = 1

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 G-Assist")
    st.markdown("AI Ticket Classifier")
    st.markdown("---")
    
    st.markdown("**เมนูหลัก**")
    
    if st.button("🔍 วิเคราะห์ปัญหา", use_container_width=True, key="nav_analyze"):
        st.session_state.page = "analyze"
        st.rerun()
    
    if st.button("📋 รายการ Ticket", use_container_width=True, key="nav_tickets"):
        st.session_state.page = "tickets"
        st.rerun()
    
    st.markdown("---")
    st.markdown("**ระบบ**")
    
    if st.button("⚙️ ตั้งค่า", use_container_width=True, key="nav_settings"):
        st.session_state.page = "settings"
        st.rerun()
    
    if st.button("📖 คู่มือ", use_container_width=True, key="nav_guide"):
        st.session_state.page = "guide"
        st.rerun()
    
    st.markdown("---")
    st.markdown("**สถานะระบบ**")
    st.success("✅ ระบบพร้อมใช้งาน")
    st.markdown(f"📊 Ticket ที่วิเคราะห์: **{len(st.session_state.history)}**")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: ANALYZE (Main Page)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "analyze":
    st.markdown("# 🔍 วิเคราะห์ปัญหาผู้เล่น")
    st.markdown("ระบบ AI จะจัดหมวดหมู่ปัญหาโดยอัตโนมัติ เพื่อส่งต่อทีมงานที่เกี่ยวข้องได้อย่างรวดเร็ว")
    
    st.markdown("---")
    
    st.markdown("## 🤖 AI Ticket Classifier")
    st.markdown("**ข้อความจากผู้เล่น**")
    st.markdown("💡 พิมพ์หรือวางข้อความที่ผู้เล่นแจ้งปัญหามา ระบบจะวิเคราะห์และจัดหมวดหมู่ให้โดยอัตโนมัติ")
    
    user_input = st.text_area(
        label="ข้อความจากผู้เล่น",
        placeholder="ตัวอย่าง: เติมเงินเข้าเกมผ่านพร้อมเพย์แล้วเพชรไม่ขึ้นครับ...",
        height=140,
        label_visibility="collapsed",
    )
    
    st.markdown("รองรับภาษาไทยและภาษาอังกฤษ · ความยาวแนะนำ 10–500 ตัวอักษร")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_btn = st.button("🔍 วิเคราะห์ปัญหา", use_container_width=True, key="analyze")
    with col2:
        clear_btn = st.button("🗑️ ล้าง", use_container_width=True, key="clear")
    
    # Logic
    if clear_btn:
        st.session_state.last_result = None
        st.rerun()
    
    if analyze_btn:
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ กรุณาพิมพ์ข้อความก่อนกดวิเคราะห์")
        else:
            if model is None or vectorizer is None:
                st.error("❌ ไม่พบไฟล์ model.pkl หรือ vectorizer.pkl ในโฟลเดอร์ปัจจุบัน")
            else:
                with st.spinner("🔄 กำลังวิเคราะห์..."):
                    test_vec = vectorizer.transform([user_input])
                    prediction = model.predict(test_vec)[0]
                
                now = datetime.datetime.now()
                ticket_id = f"#T-{st.session_state.ticket_counter:04d}"
                st.session_state.ticket_counter += 1
                st.session_state.last_result = {
                    "prediction": prediction,
                    "ticket_id": ticket_id,
                    "time": now.strftime("%H:%M:%S"),
                    "date": now.strftime("%d/%m/%Y"),
                    "msg": user_input[:60] + ("..." if len(user_input) > 60 else ""),
                    "full_msg": user_input,
                }
                
                st.session_state.history.insert(0, {
                    "id": ticket_id,
                    "msg": st.session_state.last_result["msg"],
                    "full_msg": user_input,
                    "category": prediction,
                    "status": "open",
                    "time": now.strftime("%H:%M:%S"),
                    "date": now.strftime("%d/%m/%Y"),
                })
                
                st.rerun()
    
    # Show Result
    if st.session_state.last_result:
        r = st.session_state.last_result
        st.success(f"✅ วิเคราะห์สำเร็จ")
        st.markdown(f"### 📂 {r['prediction']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**🎫 Ticket ID:** {r['ticket_id']}")
        with col2:
            st.markdown(f"**🕐 เวลา:** {r['time']}")
        with col3:
            st.markdown(f"**📅 วันที่:** {r['date']}")
    
    st.markdown("---")
    
    # Recent Tickets
    if st.session_state.history:
        st.markdown("## 📋 Ticket ที่วิเคราะห์")
        
        status_label = {"done": "✅ แก้ไขแล้ว", "open": "⏳ รอดำเนินการ", "review": "🔍 กำลังตรวจสอบ"}
        
        table_data = []
        for t in st.session_state.history:
            status_txt = status_label.get(t["status"], "⏳ รอดำเนินการ")
            table_data.append({
                "Ticket ID": t['id'],
                "ข้อความ": t['msg'],
                "หมวดหมู่": t['category'],
                "สถานะ": status_txt,
                "เวลา": t['time'],
                "วันที่": t['date'],
            })
        
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("ยังไม่มีข้อมูล Ticket ที่วิเคราะห์")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: TICKETS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "tickets":
    st.markdown("# 📋 รายการ Ticket ทั้งหมด")
    st.markdown("ดูรายละเอียดของ Ticket ทั้งหมดที่ได้รับการวิเคราะห์")
    
    if not st.session_state.history:
        st.info("ยังไม่มีข้อมูล Ticket ที่วิเคราะห์")
    else:
        # Filter
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            filter_status = st.selectbox("🔍 กรองตามสถานะ", ["ทั้งหมด", "รอดำเนินการ", "กำลังตรวจสอบ", "แก้ไขแล้ว"], key="filter_status")
        with col2:
            # Get unique categories from history
            categories = sorted(set([t["category"] for t in st.session_state.history]))
            filter_category = st.selectbox("📂 กรองตามหมวดหมู่", ["ทั้งหมด"] + categories, key="filter_category")
        with col3:
            sort_by = st.selectbox("📊 เรียงลำดับ", ["ล่าสุดก่อน", "เก่าที่สุดก่อน", "ID"], key="sort_by")
        
        st.markdown("---")
        
        # Filter logic
        filtered_history = st.session_state.history.copy()
        
        status_map = {"รอดำเนินการ": "open", "กำลังตรวจสอบ": "review", "แก้ไขแล้ว": "done"}
        if filter_status != "ทั้งหมด":
            filtered_history = [t for t in filtered_history if t["status"] == status_map.get(filter_status)]
        
        if filter_category != "ทั้งหมด":
            filtered_history = [t for t in filtered_history if t["category"] == filter_category]
        
        if sort_by == "เก่าที่สุดก่อน":
            filtered_history = filtered_history[::-1]
        elif sort_by == "ID":
            filtered_history = sorted(filtered_history, key=lambda x: x["id"])
        
        st.markdown(f"## ผลลัพธ์: {len(filtered_history)} รายการ")
        
        status_label = {"done": "✅ แก้ไขแล้ว", "open": "⏳ รอดำเนินการ", "review": "🔍 กำลังตรวจสอบ"}
        
        table_data = []
        for t in filtered_history:
            status_txt = status_label.get(t["status"], "⏳ รอดำเนินการ")
            table_data.append({
                "Ticket ID": t['id'],
                "ข้อความ": t['msg'],
                "หมวดหมู่": t['category'],
                "สถานะ": status_txt,
                "เวลา": t['time'],
                "วันที่": t['date'],
            })
        
        if table_data:
            st.dataframe(table_data, use_container_width=True, hide_index=True)
        else:
            st.info("ไม่พบ Ticket ที่ตรงกับเงื่อนไข")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "settings":
    st.markdown("# ⚙️ ตั้งค่า")
    st.markdown("จัดการการตั้งค่าระบบ G-Assist")
    
    st.markdown("## 🎮 ตั้งค่าทั่วไป")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("ชื่อระบบ", value="G-Assist", disabled=True)
    with col2:
        st.text_input("เวอร์ชัน", value="3.0.0", disabled=True)
    
    st.toggle("เปิดใช้งานการแจ้งเตือน", value=True)
    st.toggle("เปิดใช้งานโหมดมืด", value=True)
    
    st.markdown("---")
    
    st.markdown("## 🤖 ตั้งค่า AI Model")
    
    st.slider("ความเชื่อมั่นในการทำนาย (Confidence)", 0.0, 1.0, 0.75)
    st.selectbox("เลือก Model", ["NLP v3.0 (ปัจจุบัน)", "NLP v2.1", "NLP v2.0"])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: GUIDE
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "guide":
    st.markdown("# 📖 คู่มือการใช้งาน")
    st.markdown("วิธีการใช้งาน G-Assist Ticket Classification System")
    
    st.markdown("## 🚀 เริ่มต้นใช้งาน")
    
    st.markdown("""
    **ขั้นตอนที่ 1: วิเคราะห์ปัญหา**
    
    ไปที่หน้า "วิเคราะห์" แล้วพิมพ์ข้อความปัญหาจากผู้เล่น ระบบจะวิเคราะห์โดยอัตโนมัติ
    
    **ขั้นตอนที่ 2: ดูผลลัพธ์**
    
    ระบบจะแสดงหมวดหมู่ที่เหมาะสม พร้อม Ticket ID สำหรับติดตามปัญหา
    
    **ขั้นตอนที่ 3: จัดการ Ticket**
    
    ไปที่หน้า "รายการ Ticket" เพื่อดูประวัติและจัดการ Ticket ทั้งหมด
    """)
    
    st.markdown("---")
    
    st.markdown("## ❓ คำถามที่พบบ่อย")
    
    st.markdown("""
    **Q: ระบบรองรับภาษาอะไรบ้าง?**
    
    A: ระบบรองรับภาษาไทยและภาษาอังกฤษ
    
    **Q: ความแม่นยำของระบบเป็นเท่าไหร่?**
    
    A: ระบบมีความแม่นยำขึ้นอยู่กับคุณภาพของข้อมูลที่ป้อนเข้า
    
    **Q: เวลาตอบสนองเป็นเท่าไหร่?**
    
    A: ระบบสามารถวิเคราะห์ได้ภายในเวลาน้อยกว่า 1 วินาที
    """)
    
    st.markdown("---")
    
    st.markdown("## ⚡ เคล็ดลับการใช้งาน")
    
    st.markdown("""
    - **กรองข้อมูล:** ใช้ระบบ Filter ในหน้า "รายการ Ticket" เพื่อค้นหา Ticket ที่ต้องการ
    - **เรียงลำดับ:** ปรับการเรียงลำดับ Ticket ตามความต้องการ
    - **ตั้งค่าระบบ:** ปรับการตั้งค่าใน "ตั้งค่า" เพื่อเพิ่มประสิทธิภาพการใช้งาน
    """)
