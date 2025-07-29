import os
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# === Đường dẫn tuyệt đối ===

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_file = os.path.join(BASE_DIR, "models", "vinallama-7b-chat_q5_0.gguf")
embedding_file = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2-f16.gguf")
vector_db_path = os.path.join(BASE_DIR, "vectorstores", "db_faiss")


# === Load mô hình LLM ===
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# === Tạo PromptTemplate ===
def creat_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# === Tạo QA Chain ===
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return llm_chain

# === Đọc FAISS vector DB ===
def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file=embedding_file)
    db = FAISS.load_local(
        vector_db_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# === Khởi chạy ===
if __name__ == "__main__":
    # Bước 1: Load DB & LLM
    db = read_vectors_db()
    llm = load_llm(model_file)

    # Bước 2: Tạo Prompt
    template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.

{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
    prompt = creat_prompt(template)

    # Bước 3: Tạo chain và đặt câu hỏi
    llm_chain = create_qa_chain(prompt, llm, db)

    question = "LoRa có thể duy trì kết nối và chia sẻ dữ liệu trong thời gian lên đến bao nhiêu năm ?"
    response = llm_chain.invoke({"query": question})
    print("\n📌 Kết quả trả lời:")
    print(response)
