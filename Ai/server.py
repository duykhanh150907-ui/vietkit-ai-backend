from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Khởi tạo Web Server
app = FastAPI()

# Cho phép trang web HTML giao tiếp với Server Python mà không bị chặn bảo mật (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Tải mô hình AI (Chỉ tải 1 lần khi bật server)
print("Đang khởi động não bộ AI...")
model = SentenceTransformer('keepitreal/vietnamese-sbert')

# 1. Bổ sung thêm trường "link" vào cơ sở dữ liệu
kits = [
    {
        "name": "Đông Hồ", 
        "desc": "Tranh lợn âm dương Đông Hồ, con lợn béo tốt, gà đàn, biểu tượng của sự sung túc, no đủ và nảy nở.",
        "link": "#kit-collection" # Sau này bạn có thể thay bằng link trang chi tiết, VD: "/dong-ho.html"
    },
    {
        "name": "Bát Tràng", 
        "desc": "Họa tiết hoa sen xanh trên nền gốm Bát Tràng, con lợn đất nung, mang vẻ đẹp thanh tao, thuần khiết và truyền thống.",
        "link": "#kit-collection"
    },
    {
        "name": "Sơn Mài", 
        "desc": "Tranh sơn mài phong cảnh đồng quê, màu vàng son rực rỡ, lộng lẫy.",
        "link": "#kit-collection"
    }
]

# Chuyển đổi dữ liệu thành ma trận sẵn
kit_vectors = model.encode([k["desc"] for k in kits])
print("Server AI đã sẵn sàng lắng nghe!")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    user_query = request.message.lower()
    
    # Xử lý câu chào hỏi cơ bản
    greetings = ["alo", "chào", "hi", "hello", "xin chào", "ê", "có ai không"]
    if any(word in user_query for word in greetings) and len(user_query) < 15:
        return {"reply": "Dạ alo! Trợ lý AI Việt Kit đang nghe đây. Bạn đang muốn tìm kiếm họa tiết hay tài liệu văn hóa gì ạ? 😊"}
    
    # Chuyển câu hỏi thành vector
    query_vector = model.encode([request.message])
    scores = cosine_similarity(query_vector, kit_vectors)[0]
    
    # 2. Lọc ra TẤT CẢ các kết quả có độ khớp trên mức cho phép (ví dụ: > 0.3)
    matched_kits = []
    threshold = 0.3 
    
    for i, score in enumerate(scores):
        if score > threshold:
            matched_kits.append((kits[i], score))
            
    # Sắp xếp kết quả từ cao xuống thấp
    matched_kits.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Tạo câu trả lời dạng HTML chứa Link
    if len(matched_kits) > 0:
        reply = "Tôi đã tìm thấy các bộ Kit có chứa họa tiết bạn cần. Bạn bấm vào link để xem chi tiết nhé:<br>"
        for kit, _ in matched_kits:
            # Gắn thẻ <a> để tạo link click được, xài class Tailwind cho đẹp
            reply += f"<br>✨ <a href='{kit['link']}' class='font-inter font-bold text-[#8C6B5D] hover:text-[#6E2C24] hover:underline transition-colors'>{kit['name']}</a>"
    else:
        reply = "Chà, ý tưởng này thú vị quá nhưng kho dữ liệu của tôi chưa có bộ Kit nào thực sự khớp. Bạn miêu tả rõ hơn một chút về màu sắc hay chất liệu được không?"
        
    return {"reply": reply}