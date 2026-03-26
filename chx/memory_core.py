import asyncio
import os
import uuid
import pandas as pd
import chromadb
import httpx
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
from tqdm.asyncio import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config
from embedding import EmbeddingService



# ==========================================
# 1. 核心服务：支持批量的 Embedding Service
# ==========================================
class EmbeddingService:
    def __init__(self, api_key: str, api_url: str, model: str, dimension: int):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.dimension = dimension

    def get_embeddings_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        核心优化：批量发送请求，不再一条一条发
        """
        if not texts:
            return []

        # 简单清洗
        clean_texts = [str(t).replace('\n', ' ').strip() for t in texts]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建批量请求体
        payload = {
            "model": self.model,
            "input": clean_texts
        }

        for attempt in range(max_retries):
            try:
                # 批量请求通常耗时稍长，设置60秒超时
                with httpx.Client(verify=False, timeout=60.0) as client:
                    response = client.post(self.api_url, json=payload, headers=headers)

                    if response.status_code != 200:
                        print(f"【API 报错】状态码: {response.status_code}, 内容: {response.text}")
                        # 如果是 429 (Too Many Requests)，多睡一会
                        if response.status_code == 429:
                            time.sleep(5)
                            continue
                        raise Exception(f"API Error: {response.status_code}")

                    data = response.json()

                    # 提取数据：确保按顺序返回
                    # OpenAI 格式返回的数据里有 index，保险起见我们按 index 排序
                    data_items = data.get('data', [])
                    data_items.sort(key=lambda x: x['index'])

                    embeddings = [item['embedding'] for item in data_items]

                    # 简单校验数量
                    if len(embeddings) != len(texts):
                        print(f"警告：请求了 {len(texts)} 条，返回了 {len(embeddings)} 条")

                    return embeddings

            except Exception as e:
                print(f"批量请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 * (attempt + 1))

        # 如果全部失败，返回空列表（外层会处理）
        return []


# ==========================================
# 2. 异步生成器 (已简化)
# ==========================================
class AsyncEmbeddingGenerator:
    def __init__(self):
        self.service = EmbeddingService(
            api_key=Config.EMBEDDING_API_KEY,
            api_url=Config.EMBEDDING_API_URL,
            model=Config.EMBEDDING_MODEL,
            dimension=Config.EMBEDDING_DIMENSION
        )

    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        在异步环境中调用同步的批量接口
        """
        loop = asyncio.get_running_loop()
        # 放到线程池里去跑，防止阻塞主线程
        result = await loop.run_in_executor(
            None,
            lambda: self.service.get_embeddings_batch(texts)
        )
        return result


# ==========================================
# 3. 上下文感知记忆系统 (逻辑修正)
# ==========================================
class AdvancedMemorySystem:
    def __init__(self, collection_name: str = "ex_gf_contextual_memory"):
        # 这里的 path 建议使用绝对路径，或者确保文件夹存在
        self.client = chromadb.PersistentClient(path="./save/memory_advanced")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = AsyncEmbeddingGenerator()

    def _preprocess_data(self, df: pd.DataFrame) -> List[Dict]:
        print("正在进行数据清洗与上下文构建...")

        # --- 健壮性修复：去除列名的空格 ---
        df.columns = df.columns.str.strip()

        # 检查必要的列是否存在
        required_cols = ['消息类型', '内容', '发送者身份']
        for col in required_cols:
            if col not in df.columns:
                print(f"【错误】CSV 中缺少列: {col}。当前列名: {df.columns.tolist()}")
                return []

        # 筛选文本消息
        df = df[df['消息类型'] == '文本消息'].copy()
        df['内容'] = df['内容'].astype(str).fillna("")

        processed_docs = []
        rows = df.to_dict('records')

        # 简单的滑动窗口处理
        for i in range(len(rows) - 1):
            curr_msg = rows[i]
            next_msg = rows[i + 1]

            content_curr = str(curr_msg['内容']).strip()
            content_next = str(next_msg['内容']).strip()

            # 过滤太短的内容（节省 Token 且提高质量）
            if len(content_curr) < 2 or len(content_next) < 2:
                continue

            # 身份判断 logic
            sender_curr = str(curr_msg.get('发送者身份', '')).strip()
            sender_next = str(next_msg.get('发送者身份', '')).strip()

            # 策略A：我问 -> 她答 (对话对)
            if sender_curr == "我" and sender_next == "盒子":
                timestamp = str(curr_msg.get('时间', datetime.now().isoformat()))
                processed_docs.append({
                    "vector_text": content_curr,  # 用我的话去搜
                    "storage_text": f"User: {content_curr}\nGF: {content_next}",  # 存整个对话
                    "metadata": {
                        "type": "dialogue_pair",
                        "timestamp": timestamp,
                        "original_user_query": content_curr,
                        "gf_reply": content_next
                    },
                    "id": str(uuid.uuid4())
                })

            # 策略B：她说的长句子 (事实记忆)
            elif sender_curr == "盒子" and len(content_curr) > 15:
                processed_docs.append({
                    "vector_text": content_curr,
                    "storage_text": f"GF Memory: {content_curr}",
                    "metadata": {
                        "type": "fact_memory",
                        "timestamp": str(curr_msg.get('时间', '')),
                        "gf_reply": content_curr
                    },
                    "id": str(uuid.uuid4())
                })

        print(f"构建完成，共生成 {len(processed_docs)} 条上下文记忆片段。")
        return processed_docs

    async def ingest_excel(self, file_path: str, batch_size: int = 150):  # batch_size 可以设为 50 或 100
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return

        print(f"正在读取 CSV: {file_path}")
        try:
            # 尝试读取，处理中文乱码
            df = pd.read_csv(file_path, skiprows=3, encoding='gb18030', encoding_errors='replace')
        except Exception as e:
            print(f"读取 CSV 失败: {e}")
            return

        documents = self._preprocess_data(df)
        total_docs = len(documents)

        if total_docs == 0:
            print("没有提取到有效数据，程序终止。")
            return

        print(f"开始向量化处理，共 {total_docs} 条数据，每次打包处理 {batch_size} 条...")

        # 进度条
        for i in tqdm(range(0, total_docs, batch_size), desc="Ingesting Memory"):
            # 1. 切片（取出这一批数据）
            batch = documents[i: i + batch_size]

            # 2. 准备文本列表
            texts_to_embed = [doc['vector_text'] for doc in batch]
            ids = [doc['id'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            storage_texts = [doc['storage_text'] for doc in batch]

            # 3. 批量调用 API (关键提速步骤)
            embeddings = await self.embedder.generate_batch(texts_to_embed)

            # 4. 检查结果并存入数据库
            valid_ids = []
            valid_embeddings = []
            valid_metadatas = []
            valid_documents = []

            if embeddings and len(embeddings) == len(batch):
                for j, emb in enumerate(embeddings):
                    # 确保向量非空且长度正确
                    if emb and len(emb) > 0:
                        valid_ids.append(ids[j])
                        valid_embeddings.append(emb)
                        valid_metadatas.append(metadatas[j])
                        valid_documents.append(storage_texts[j])

                # 批量写入 ChromaDB
                if valid_ids:
                    self.collection.add(
                        ids=valid_ids,
                        embeddings=valid_embeddings,
                        metadatas=valid_metadatas,
                        documents=valid_documents
                    )
            else:
                print(f"批次 {i} 处理失败或返回为空")

        print("\n🎉 记忆注入完成！")

    async def search(self, query: str, top_k: int = 3) -> List[Dict]:
        # 搜索时也用 batch 接口，虽然只有一条
        query_embedding_list = await self.embedder.generate_batch([query])

        if not query_embedding_list or not query_embedding_list[0]:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding_list,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                doc = results['documents'][0][i]
                dist = results['distances'][0][i]

                formatted_results.append({
                    "content": doc,
                    "type": meta.get('type', 'unknown'),
                    "reply_reference": meta.get('gf_reply', ''),
                    "score": 1 - dist
                })
        return formatted_results


# ==========================================
# 4. 主程序
# ==========================================
async def main():
    memory_system = AdvancedMemorySystem()

    # 你的文件路径
    excel_path = r"./data/盒子.csv"

    # 执行注入
   # await memory_system.ingest_excel(excel_path, batch_size=150)

    # 测试搜索
    test_query = "你怎么不干了"
    print(f"\n🔍 测试搜索: '{test_query}'")
    matches = await memory_system.search(test_query, top_k=3)

    for idx, match in enumerate(matches):
        print(f"--- 结果 {idx + 1} (相似度: {match['score']:.4f}) ---")
        print(f"记忆: {match['content']}")


if __name__ == "__main__":
    import sys

    # Windows下 asyncio 的策略设置
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())