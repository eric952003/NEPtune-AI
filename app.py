import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_engine import BiotechRAG  # 引入我們之前寫好的 RAG 檢索引擎

# ------------------------------
# 1. 網頁基本設定與側邊欄
# ------------------------------
st.set_page_config(page_title="Qwen 生技 AI RAG 助理", page_icon="🧬", layout="centered")
st.title("🧬 專屬生技 AI 智能助理 (Qwen 2.5 + RAG 旗艦版)")
st.markdown("結合了 **Qwen 2.5-1.5B 開源旗艦大模型** 與 **ChromaDB 向量檢索**，提供業界最高水準的生技專業問答！")

with st.sidebar:
    st.header("⚙️ 生成參數控制")
    n_results = st.slider("檢索知識片段數 (RAG Top-k)", min_value=1, max_value=5, value=2, step=1)
    temperature = st.slider("創造力 (Temperature)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
    max_tokens = st.slider("最大回覆長度 (Max Tokens)", min_value=50, max_value=500, value=250, step=50)
    st.info("💡 提示：輸入你想了解的生技主題（例如：PCR、基因工程、酵素），系統會自動檢索資料庫並由 Qwen 大模型為你深度解答。")

# ------------------------------
# 2. 快取載入 Qwen 大模型與 RAG 引擎
# ------------------------------
@st.cache_resource
def load_heavy_resources():
    print("📥 正在載入 RAG 檢索引擎...")
    rag = BiotechRAG(dataset_path="dataset.txt")
    
    print("📥 正在載入 Qwen 2.5 開源大模型...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 根據硬體自動選擇用 CPU 或半精度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 強制使用 CPU 載入，避開雲端無 GPU 的報錯
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # CPU 模式下使用 float32 最穩定
        trust_remote_code=True
    ).to("cpu")
    )
    return tokenizer, model, rag

tokenizer, model, rag = load_heavy_resources()

# ------------------------------
# 3. 對話介面互動邏輯
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入生技主題，例如：PCR 的原理、基因工程..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Qwen 正在檢索知識庫並深度思考中..."):
            # 1. 透過 RAG 檢索最相關的知識片段
            retrieved_chunks = rag.search(prompt, n_results=n_results)
            
            # 2. 組合檢索到的上下文作為 Prompt 的一部份 (RAG 核心)
            context_text = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])
            
            system_prompt = (
                "你是一個專業的生物科技專家助理。請根據以下提供的「參考知識庫」，"
                "以精準、專業且流暢的繁體中文回答使用者問題。如果知識庫中沒有直接答案，請發揮你的專業知識進行補充。\n\n"
                f"【參考知識庫】：\n{context_text}"
            )
            
            # 3. 透過 Qwen 構建對話訊息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
            
            # 4. 讓 Qwen 大模型生成回答
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 5. 組合最終輸出格式 (展示檢索依據 + 大模型回答)
            final_output = f"### 📚 知識庫檢索依據 (RAG)：\n{context_text}\n\n### 🤖 Qwen 專業生技專家解答：\n{response_text}"
            
            st.markdown(final_output)
    
    st.session_state.messages.append({"role": "assistant", "content": final_output})
