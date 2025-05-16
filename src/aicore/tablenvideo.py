import tempfile
from airesource.config import LLMNAME_GENERATE
import ollama


def answer_figure_with_gemma3(image, user_query: str, user_info: dict, context: str):
    """
    Use Gemma3 Vision LLM to answer a user query about a figure, personalized by user_info/context.
    """
    # Save image to temp file for ollama
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        prompt = f"""
    Bạn là một Chuyên gia Giải thích AI, có khả năng phân tích hình ảnh và trình bày lại nội dung một cách phù hợp với trình độ của sinh viên.
    
    Mục tiêu: Trả lời câu hỏi của sinh viên về hình ảnh dưới đây, dựa trên Ngữ cảnh (đánh giá của giáo viên) và điểm số của sinh viên trong môn học liên quan để điều chỉnh câu trả lời phù hợp nhất có thể đối với sinh viên.

    Ví dụ về cách điều chỉnh:
    --------------------------
    Ví dụ 1 (Sinh viên điểm cao môn liên quan):
    Câu hỏi: Giải thích nguyên lý hoạt động của bộ nhớ RAM.
    Câu trả lời (chuyên sâu): RAM (Random Access Memory) là bộ nhớ truy cập ngẫu nhiên... 
    
    Ví dụ 2 (Sinh viên điểm thấp môn liên quan):
    Câu hỏi: Giải thích nguyên lý hoạt động của bộ nhớ RAM.
    Câu trả lời (đơn giản): RAM giống như bàn làm việc tạm thời của máy tính... 
    --------------------------
    
    Các bước tạo câu trả lời:
    1. **Xác định môn học liên quan** qua Ngữ cảnh hoặc từ Câu hỏi.
    2. **Kiểm tra đánh giá/điểm số** trong Ngữ cảnh (cao >= 7.0 => chuyên sâu, thấp < 7.0 => đơn giản).
    3. **Soạn thảo** chỉ 1 phiên bản giải thích phù hợp.
    4. **Đảm bảo**: toàn bằng tiếng Việt, không nhắc đến các mức độ khác, chỉ dùng thông tin trong Tài liệu.
        
    Và dựa trên ngữ cảnh:
    {context}
    
    Và thông tin người dùng:
    {user_info}
    
    Hãy trả lời câu hỏi: {user_query}
    
    Hãy trả lời bằng tiếng Việt, điều chỉnh độ chi tiết/phức tạp dựa trên điểm số của sinh viên trong môn học liên quan. Chỉ cung cấp một phiên bản phù hợp với trình độ người dùng.
    """
    res = ollama.chat(
        model=LLMNAME_GENERATE,
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [tmp.name]
            }
        ]
    )
    return res['message']['content']


def answer_table_with_gemma3(model, table_text: str, user_query: str, user_info: dict, context: str):
    """
    Use Gemma3 LLM to answer a user query about a table, personalized by user_info/context.
    """
    prompt = f"""
    Bạn là một Chuyên gia Giải thích AI, có khả năng phân tích bảng dữ liệu và trình bày lại nội dung một cách phù hợp với trình độ của sinh viên.
    
    Mục tiêu: Trả lời câu hỏi của sinh viên về bảng dữ liệu dưới đây, dựa trên Ngữ cảnh (đánh giá của giáo viên) và điểm số của sinh viên trong môn học liên quan để điều chỉnh câu trả lời phù hợp nhất có thể đối với sinh viên.

    Ví dụ về cách điều chỉnh:
    --------------------------
    Ví dụ 1 (Sinh viên điểm cao môn liên quan):
    Câu hỏi: Giải thích nguyên lý hoạt động của bộ nhớ RAM.
    Câu trả lời (chuyên sâu): RAM (Random Access Memory) là bộ nhớ truy cập ngẫu nhiên... 
    
    Ví dụ 2 (Sinh viên điểm thấp môn liên quan):
    Câu hỏi: Giải thích nguyên lý hoạt động của bộ nhớ RAM.
    Câu trả lời (đơn giản): RAM giống như bàn làm việc tạm thời của máy tính... 
    --------------------------
    
    Các bước tạo câu trả lời:
    1. **Xác định môn học liên quan** qua Ngữ cảnh hoặc từ Câu hỏi.
    2. **Kiểm tra đánh giá/điểm số** trong Ngữ cảnh (cao >= 7.0 => chuyên sâu, thấp < 7.0 => đơn giản).
    3. **Soạn thảo** chỉ 1 phiên bản giải thích phù hợp.
    4. **Đảm bảo**: toàn bằng tiếng Việt, không nhắc đến các mức độ khác, chỉ dùng thông tin trong Tài liệu.
    
    Ngữ cảnh (Đánh giá của giáo viên):
    {context}
    
    Thông tin người dùng:
    {user_info}
    
    Nội dung bảng:
    {table_text}
    
    Câu hỏi: {user_query}
    
    Hãy trả lời bằng tiếng Việt, điều chỉnh độ chi tiết/phức tạp dựa trên điểm số của sinh viên trong môn học liên quan. Chỉ cung cấp một phiên bản phù hợp với trình độ người dùng.
    """
    return model.run(prompt)["replies"][0]
