import asyncio
from typing import List
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from datetime import datetime
import uuid
from embedding import EmbeddingService
from config import Config
import pandas as pd
import os


# ... (APIEmbeddingFunction 部分保持不变) ...
class APIEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.embedding_service = EmbeddingService(
            api_key=Config.EMBEDDING_API_KEY,
            api_url=Config.EMBEDDING_API_URL,
            model=Config.EMBEDDING_MODEL,
            dimension=Config.EMBEDDING_DIMENSION
        )

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                embedding = self.embedding_service.get_embedding(text)
                if embedding is None:
                    embedding = [0.0] * Config.EMBEDDING_DIMENSION
                embeddings.append(embedding)
            except Exception as e:
                print(f"获取embedding时出错: {e}")
                embedding = [0.0] * Config.EMBEDDING_DIMENSION
                embeddings.append(embedding)
        return embeddings


class ExcelToVectorDB:
    def __init__(self, collection_name: str = "gf_memory"):
        self.client = chromadb.PersistentClient(path="save/memory")  # 修改为持久化客户端
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=APIEmbeddingFunction()
        )

    def process_excel(self, file_path: str, batch_size: int = 100):
        """
        处理 WeFlow 导出的聊天记录
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # --- 修改点 1: 读取设置 ---
            # skiprows=4 跳过元数据，encoding处理中文
            print(f"正在读取文件: {file_path}")
            df = pd.read_csv(file_path, skiprows=4, encoding='utf-8-sig')

            # --- 修改点 2: 数据清洗 ---
            # 只保留“文本消息”
            df = df[df['消息类型'] == '文本消息'].copy()
            # 确保内容列是字符串
            df['内容'] = df['内容'].astype(str).fillna("")

            # --- 修改点 3: 逻辑升级（建立 QA 对记忆） ---
            # 我们索引【我】说的话，关联【盒子】的回答。这样你问AI时，它能找到盒子当时的反应。
            documents = []
            metadatas = []
            ids = []

            rows = df.to_dict('records')
            for i in range(len(rows) - 1):
                curr = rows[i]
                next_msg = rows[i + 1]

                # 如果当前是我说话，下一句是盒子回答
                if str(curr['发送者身份']).strip() == "我" and str(next_msg['发送者身份']).strip() == "盒子":
                    documents.append(curr['内容'])  # 索引我的“提问”
                    metadatas.append({
                        "gf_response": next_msg['内容'],  # 存入她的“回答”
                        "time": str(curr.get('时间', '')),
                        "source": "wechat_chat"
                    })
                    ids.append(str(uuid.uuid4()))

            total_records = len(documents)
            print(f"成功构建对话对: {total_records} 组")

            # 批量存入数据库
            for i in range(0, total_records, batch_size):
                end = min(i + batch_size, total_records)
                self.collection.add(
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
                print(f"进度: {end}/{total_records}")

            print("✅ 向量数据库构建完成！")

        except Exception as e:
            print(f"❌ 处理文件时出错: {str(e)}")

    def search_similar(self, query: str, n_results: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas']
        )
        return results


if __name__ == "__main__":
    async def main():
        processor = ExcelToVectorDB()

        # --- 修改点 4: 路径和列名配置 ---
        excel_path = r"./data/盒子.csv"

        # 第一次运行请解开下行的注释
        #processor.process_excel(excel_path)

        # 测试搜索
        query = "你在干嘛"
        print(f"\n🔍 模拟用户提问: '{query}'")
        results = processor.search_similar(query, n_results=3)

        print("\n🧠 检索到的历史记忆:")
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            print(f"\n当你说: '{doc}' 时")
            print(f"盒子曾回复: '{meta['gf_response']}'")


    asyncio.run(main())