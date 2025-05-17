from enum import Enum
from aicore.airesource.prompt import prompt_getemotion, prompt_emolize
import random

class Emotion(Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


def get_emo(model, query):
    prompt = prompt_getemotion.format(query=query)
    res = model.run(prompt)["replies"][0]

    if "Positive" in res:
        return Emotion.POSITIVE
    if "Negative" in res:
        return Emotion.NEGATIVE
    elif "Neutral" in res:
        return Emotion.NEUTRAL


def get_emodata(emo: Emotion):
    if emo == Emotion.POSITIVE:
        return "Cảm xúc: Vui vẻ, Tích cực", "Vui vẻ, nhiệt tình, tích cực",  ["😄", "😂", "🤣", "😅", "😊", "🥰", "😍", "😆", "😇", "😁"]
    if emo == Emotion.NEGATIVE:
        return "Cảm xúc: Buồn bã, Tiêu cực", "Cảm thông, nhẹ nhàng, hiểu biết", ["😬", "😗", "🤒", "😯", "😕", "😖", "😱", "😨", "😰", "😳"]
    elif emo == Emotion.NEUTRAL:
        return "Cảm xúc: Bình thường", "Trung lập, khách quan",  ["👀", "🤗", "🤔", "🧐", "😎", "🤯", "👻", "👽", "🤖", "👾"]

def emolize(model, message, emodata):
    nhan_phu, tong, emoji = emodata
    emoji = random.choice(emoji)

    prompt = prompt_emolize.format(
        nhan_phu=nhan_phu, tong=tong, emoji=emoji, message=message)
    return model.run(prompt)["replies"][0]


def fulll_emolize(model, query, message):
    emodata = get_emodata(get_emo(model, query))
    message = emolize(model, message, emodata)

    return message


def personalized_answer_from_summary(summary: str, user_query: str, user_info: dict):
    raise NotImplementedError("Just to make sure no code use this")
    
    """
    Use Gemma3 LLM to generate a personalized answer based on the summary, user query, and student transcript.
    """
    rubric = """
    Bạn là một giáo viên siêu nghiêm khắc, người đánh giá hiệu suất của học sinh với độ chính xác không ngừng nghỉ.
    Công việc của bạn là chấm điểm từng môn học dựa trên điểm số của học sinh và cung cấp phản hồi nghiêm khắc và chi tiết.
    Sử dụng các quy tắc và mô tả sau để đánh giá học sinh:

    Tiêu chí chấm điểm:
    - Điểm từ 9.0 đến 10.0: Tuyệt vời! - Bạn đã thể hiện sự nắm vững kiến thức và kỹ năng xuất sắc trong môn học này. Nỗ lực và sự chính xác của bạn rất đáng khen ngợi. Hãy tiếp tục duy trì phong độ tuyệt vời này và thử thách bản thân với những kiến thức sâu hơn nhé!
    - Điểm từ 8.0 đến 8.9: Rất tốt! - Bạn hiểu bài rất tốt và thể hiện năng lực ấn tượng. Chỉ còn một vài điểm nhỏ nữa là đạt đến mức hoàn hảo. Cùng xem lại những chi tiết nhỏ này để giúp bạn hoàn thiện hơn nữa kiến thức và kỹ năng của mình.
    - Điểm từ 7.0 đến 7.9: Tốt! - Bạn đã nắm được những kiến thức và kỹ năng cơ bản, quan trọng của môn học. Đây là một nền tảng tốt. Để hiểu sâu sắc hơn, bạn có thể tập trung thêm vào việc củng cố [chỉ ra lĩnh vực cụ thể nếu có thể] và luyện tập thêm. Đừng ngần ngại đặt câu hỏi về những phần bạn còn băn khoăn.
    - 6.0 đến 6.9: Khá! - Có sự cố gắng rõ rệt và bạn đã nắm bắt được một phần kiến thức. Tuy nhiên, sự hiểu biết cần được đào sâu hơn để thực sự vững vàng. Hãy cùng xác định những phần kiến thức cần củng cố thêm. Một số gợi ý cụ thể có thể giúp bạn tiến bộ nhanh hơn đấy.
    - Điểm từ 5.0 đến 5.9: Cần cố gắng hơn! - Có vẻ như bạn đang gặp một số thử thách với các khái niệm cốt lõi của môn học này. Không sao cả, ai cũng có lúc gặp khó khăn. Điều quan trọng là xác định được những điểm bạn chưa thực sự hiểu rõ. Hãy bắt đầu từ đó, chúng ta có thể cùng nhau xây dựng một kế hoạch học tập phù hợp để bạn cải thiện nhé. 
    - Điểm dưới 5.0: Kết quả này cho thấy phương pháp học hiện tại có thể chưa phù hợp nhất với bạn trong môn học này. Đây là một tín hiệu để chúng ta cùng nhìn lại. Đừng nản lòng! Đây là cơ hội để khám phá những cách tiếp cận mới hiệu quả hơn. Hãy thử bắt đầu lại từ những kiến thức nền tảng nhất và tìm kiếm sự hỗ trợ khi cần thiết. Luôn có cách để tiến bộ!
    
    Quy tắc đánh giá:
    1. Cung cấp phản hồi cho từng môn học mà học sinh được chấm điểm.
    2. Tập trung vào cả điểm mạnh và điểm yếu, không để lại sự mơ hồ về các lĩnh vực cần cải thiện.
    3. Nếu học sinh đạt dưới 7.0 ở bất kỳ môn nào, hãy nêu rõ điều này như một vấn đề cần chú ý ngay lập tức.
    4. Nếu học sinh đạt dưới 5.0 ở bất kỳ môn nào, nhấn mạnh điều này như một sự thất bại và đề nghị các hành động khắc phục mạnh mẽ.
    

    Dưới đây là dữ liệu của học sinh:
    {user_info}
    """

    prompt = """
    Dưới đây là đánh giá của học sinh:
    {ranking}
    
    Câu hỏi của học sinh:
    {user_query}
    
    Tóm tắt nội dung liên quan:
    {summary}
    
    Hãy trả lời câu hỏi trên, điều chỉnh độ chi tiết/phức tạp của câu trả lời dựa trên điểm số của học sinh trong môn học liên quan: 
    - Nếu điểm cao (>=7.0), giải thích sâu sắc, chi tiết, có thể mở rộng thêm kiến thức nâng cao.
    - Nếu điểm thấp (<7.0), giải thích đơn giản, dễ hiểu, tập trung vào khái niệm cơ bản.
    
    """

    ranking = OllamaGenerator(model=llmname_generate).run(
        rubric.format(user_info=user_info))["replies"][0]
    response = OllamaGenerator(model=llmname_generate).run(prompt.format(
        ranking=ranking, user_query=user_query, summary=summary))["replies"][0]
    response = fulll_emolize(user_query, response)
    return response
