import pickle
from pythainlp.tokenize import word_tokenize

# ฟังก์ชันตัดคำ
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

# โหลดโมเดล
print("กำลังโหลดสมองของบอท...")
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ข้อสอบใหม่! (ประโยคพวกนี้ไม่มีใน dataset.csv)
test_sentences = [
    "แอดมิน โอนเงินไปแล้วแต่เพชรไม่เด้งอะ รอนานละนะ",       
    "เซิฟเน่าป่าวเนี่ย ปิงแดงเถือกเลย เล่นๆอยู่ก็หลุด",          
    "บอสด่าน 5-10 มันตีแรงจังพี่ มีสูตรหลบสกิลมันป่าว",        
    "จำพาสเวิดไม่ได้ครับ เมลก็เข้าไม่ได้ กู้ยังไงได้บ้าง",         
    "ซื้อแพ็กเกจไปแล้วหน้าจอมันค้าง พอกดรีเฟรชเงินในบัญชีโดนหักไปแล้วแต่ของไม่เข้า บัคหรือเปล่าเนี่ย"            
]

# ให้บอทลองทำนาย
print("\n--- ผลการทดสอบ ---")
test_vec = vectorizer.transform(test_sentences)
predictions = model.predict(test_vec)

# ปริ้นท์ผลออกมาดู
for text, pred in zip(test_sentences, predictions):
    print(f"💬 พิมพ์ว่า: '{text}'")
    print(f"🤖 บอทจัดหมวดให้เป็น: {pred}\n")