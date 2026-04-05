import pickle
from pythainlp.tokenize import word_tokenize

# 1. สร้างฟังก์ชันตัดคำแบบเดียวกับตอนเทรนเป๊ะๆ
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

# 2. โหลดโมเดลที่เราเทรนไว้ (ไฟล์ .pkl ต้องอยู่ในโฟลเดอร์เดียวกัน)
print("กำลังโหลดสมองของบอท...")
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 3. ลองตั้งประโยคจำลองที่ลูกค้าอาจจะพิมพ์มาจริงๆ (ภาษาแชท/พิมพ์ผิดนิดหน่อย)
test_sentences = [
    "เติมเงินแล้วไม่ได้ของ ",       # ควรทายเป็น Payment
    "เกมแลคมาก",           # ควรทายเป็น Server
    "โดนแบน",        # ควรทายเป็น Gameplay
   
]

# 4. ให้โมเดลลองทำนาย
print("\n--- ตัวอย่าง input output ---")
test_vec = vectorizer.transform(test_sentences)
predictions = model.predict(test_vec)

# 5. ปริ้นท์ผลลัพธ์ออกมาดู
for text, pred in zip(test_sentences, predictions):
    print(f"💬 Input: '{text}'")
    print(f"🤖 Output: {pred}\n")