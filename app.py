import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import jieba

# ------------------------------
# 1. 網頁基本設定與側邊欄
# ------------------------------
st.set_page_config(page_title="專屬生技 AI 助理", page_icon="🧬", layout="centered")
st.title("🧬 專屬生技 AI 助理")
st.markdown("這是一個由你自己親手打造的 Transformer 語言模型，具備基礎生物化學與分生技術知識。")

with st.sidebar:
    st.header("⚙️ 生成參數設定")
    temperature = st.slider("創造力 (Temperature)", min_value=0.1, max_value=2.0, value=1.1, step=0.1)
    top_p = st.slider("核抽樣 (Top-p)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    max_tokens = st.slider("生成長度 (Max Words)", min_value=10, max_value=300, value=100, step=10)
    st.info("💡 提示：輸入一個生技專有名詞（如：微生物、蛋白質、酵素），AI 會自動接續完成該領域的專業論述。")

# ------------------------------
# 2. 模型架構定義 (必須與訓練時完全一致)
# ------------------------------
block_size = 32
n_embd = 64
n_head = 4
n_layer = 3
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        T = x.size(1)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx):
        T = idx.size(1)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        self.eval() 
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------------------------
# 3. 資源快取與初始化 (確保網頁流暢度)
# ------------------------------
@st.cache_resource
def load_resources():
    # 讀取原本的資料集來重建字典
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    words = jieba.lcut(text)
    unique_words = sorted(list(set(words)))
    vocab_size = len(unique_words)
    
    stoi = { w:i for i,w in enumerate(unique_words) }
    itos = { i:w for i,w in enumerate(unique_words) }
    
    # 載入模型權重
    model = TransformerLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load('biotech_model_ultimate.pth', map_location=device))
    
    return model, stoi, itos

model, stoi, itos = load_resources()

# 編碼與解碼函數
encode = lambda s: [stoi[w] for w in jieba.lcut(s) if w in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ------------------------------
# 4. 對話介面互動邏輯
# ------------------------------
# 儲存對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示過去的對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收使用者輸入
if prompt := st.chat_input("請輸入開頭詞彙，例如：酵素、DNA、培養基..."):
    # 顯示使用者的輸入
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 生成回應
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            encoded_input = encode(prompt)
            if not encoded_input:
                response = "⚠️ 字典中找不到這個詞彙，請嘗試輸入資料集內存在過的生技專有名詞。"
            else:
                context = torch.tensor(encoded_input, dtype=torch.long, device=device).unsqueeze(0)
                out = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)
                response = decode(out[0].tolist())
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})