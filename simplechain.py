from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

# Đường dẫn đến model GGUF cục bộ
model_file = "C:/Users/pc/PycharmProjects/test1/models/vinallama-7b-chat_q5_0.gguf"

# Load model GGUF
def load_llm(model_path):
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        config={
            'max_new_tokens': 1024,
            'temperature': 0.01
        }
    )
    return llm

# Tạo prompt
def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

# Prompt template
template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

# Tạo pipeline
prompt = creat_prompt(template)
llm = load_llm(model_file)
chain = prompt | llm  # Đây là pipeline mới thay cho LLMChain

# Gọi chuỗi xử lý
response = chain.invoke({"question": "Một cộng một bằng mấy?"})
print(response)
