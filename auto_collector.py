import requests
import re
import jieba

# 定義你想自動蒐集的生技主題關鍵字清單
topics = ["生物科技", "基因工程", "發酵工業", "酶", "聚合酶連鎖反應"]

def fetch_wikipedia_article(title):
    """透過維基百科 API 自動抓取指定條目的內文"""
    url = "https://zh.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",      # 【已修正】補上正確的引號
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True
    }
    # 設定合法的 User-Agent 避免被伺服器阻擋
    headers = {"User-Agent": "BiotechAIBot/1.0 (Educational Project)"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        
        # 【新增防護】檢查 HTTP 狀態碼
        if response.status_code != 200:
            print(f"❌ 請求失敗，狀態碼: {response.status_code}")
            return ""
            
        # 【新增防護】確保內容不為空
        if not response.text.strip():
            print("❌ 伺服器回傳了空內容！")
            return ""
            
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            if page_id != "-1":
                return page_info.get("extract", "")
                
    except requests.exceptions.JSONDecodeError:
        print(f"❌ 解析 JSON 失敗，伺服器可能回傳了非 JSON 格式的內容。")
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")
        
    return ""

def clean_text(raw_text):
    """文字清洗器：去除不必要的空白、換行與特殊符號"""
    # 移除網址
    text = re.sub(r'http\S+', '', raw_text)
    # 將多餘的空白或連續換行符號取代為單一空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("🚀 開始執行自動化資料蒐集管線...")
accumulated_corpus = ""

for topic in topics:
    print(f"正在自動抓取主題: {topic} ...")
    raw_content = fetch_wikipedia_article(topic)
    if raw_content:
        cleaned = clean_text(raw_content)
        accumulated_corpus += cleaned + "\n"
        print(f"✅ 成功抓取並清洗，字數: {len(cleaned)}")
    else:
        print(f"⚠️ 找不到主題 {topic} 的資料或抓取失敗。")

# 將自動蒐集到的新資料自動附加到 dataset.txt 的最下方
if accumulated_corpus.strip():
    with open('dataset.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + accumulated_corpus)
    print(f"\n🎉 自動化資料擴充完成！新的生技知識已成功注入 dataset.txt。")
    print(f"目前累積新增了約 {len(accumulated_corpus)} 字的專業語料。")
else:
    print("\n⚠️ 沒有成功蒐集到任何新資料，dataset.txt 未被修改。")
