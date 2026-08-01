import chromadb
from sentence_transformers import SentenceTransformer

class BiotechRAG:
    def __init__(self, dataset_path="dataset.txt"):
        print("init: 初始化 RAG 檢索引擎與向量模型...")
        # 1. 載入輕量級的中文語意向量模型 (支援繁體中文與專業術語)
        self.encoder = SentenceTransformer('shibing624/text2vec-base-chinese')
        
        # 2. 建立本地端記憶體向量資料庫 (ChromaDB)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="biotech_kb")
        
        # 3. 載入並切分資料集
        self._load_and_vectorize(dataset_path)

    def _load_and_vectorize(self, path):
        """讀取 dataset.txt 並將其切段存入向量資料庫"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"❌ 找不到檔案: {path}")
            return

        # 簡單以句號或段落換行來切分文本 (Chunking)
        # 濾掉過短的字串
        chunks = [c.strip() for c in text.replace('\n', '。').split('。') if len(c.strip()) > 5]
        
        if not chunks:
            print("⚠️ 資料集內容過少，無法建立有效切段。")
            return

        print(f"📦 成功將文本切分為 {len(chunks)} 個知識片段，正在進行向量化...")
        
        # 將文字轉為向量
        embeddings = self.encoder.encode(chunks).tolist()
        
        # 準備 IDs 與 Metadata
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": "dataset.txt"} for _ in range(len(chunks))]
        
        # 寫入 ChromaDB 資料庫
        self.collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print("🎉 向量資料庫建置完成！")

    def search(self, query, n_results=2):
        """根據使用者的提問，檢索出最相關的知識片段"""
        query_vector = self.encoder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=n_results
        )
        
        # 取出檢索到的文本清單
        retrieved_docs = results.get("documents", [[]])[0]
        return retrieved_docs

# 簡單自我測試
if __name__ == "__main__":
    rag = BiotechRAG()
    test_query = "什麼是 PCR 的循環步驟？"
    print(f"\n🔍 測試查詢: '{test_query}'")
    matches = rag.search(test_query, n_results=2)
    for i, doc in enumerate(matches):
        print(f"  [檢索結果 {i+1}] {doc}")
