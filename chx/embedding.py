import httpx
import asyncio
from typing import List, Optional, Union
import time


class EmbeddingService:
    def __init__(self, api_key: str, api_url: str, model: str, dimension: int):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.dimension = dimension

    def get_embeddings_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        增强版：支持自动降级和空值清洗
        """
        # 1. 深度清洗：去除空字符串，防止 API 报错
        # 记录原始索引，以便后续还原顺序（虽然 embedding 通常不需要严格对应空行，但保持逻辑严谨）
        valid_texts = []
        valid_indices = []
        for idx, t in enumerate(texts):
            clean_t = str(t).replace('\n', ' ').strip()
            if clean_t:  # 只有非空才发送
                valid_texts.append(clean_t)
                valid_indices.append(idx)

        if not valid_texts:
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 尝试批量发送
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "input": valid_texts  # 发送列表
                }

                with httpx.Client(verify=False, timeout=60.0) as client:
                    response = client.post(self.api_url, json=payload, headers=headers)

                    # === 关键修改：如果遇到 400 错误，说明不支持批量，转为单条处理 ===
                    if response.status_code == 400:
                        print(f"⚠️ API 不支持批量或格式校验失败 (400)，正在自动切换为【逐条发送】模式...")
                        return self._fallback_single_processing(valid_texts, headers)

                    if response.status_code != 200:
                        print(f"【API 报错】状态码: {response.status_code}, 内容: {response.text}")
                        if response.status_code == 429:  # 限流
                            time.sleep(5)
                            continue
                        raise Exception(f"API Error: {response.status_code}")

                    # 解析成功响应
                    data = response.json()
                    data_items = data.get('data', [])

                    # 某些 API 可能不返回 index，按列表顺序假设
                    if data_items and 'index' in data_items[0]:
                        data_items.sort(key=lambda x: x['index'])

                    embeddings = [item['embedding'] for item in data_items]
                    return embeddings

            except Exception as e:
                print(f"批量请求异常: {e}")
                if attempt == max_retries - 1:
                    print("尝试自动切换为逐条发送模式...")
                    return self._fallback_single_processing(valid_texts, headers)
                time.sleep(2)

        return []

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        兼容性方法：将单条文本请求转为批量请求。
        解决 'EmbeddingService' object has no attribute 'get_embedding' 报错。
        """
        if not text or not str(text).strip():
            return None

        # 包装成列表调用批量方法
        results = self.get_embeddings_batch([text])

        # 如果有返回结果，取第一个元素（即该条文本的向量）
        if results and len(results) > 0:
            return results[0]
        return None

    def _fallback_single_processing(self, texts: List[str], headers: dict) -> List[List[float]]:
        """
        兜底方案：一条一条发
        """
        results = []
        print(f"正在使用逐条模式处理 {len(texts)} 条数据（速度较慢，但更稳定）...")

        with httpx.Client(verify=False, timeout=30.0) as client:
            for i, text in enumerate(texts):
                # 简单的重试逻辑
                for _ in range(3):
                    try:
                        payload = {
                            "model": self.model,
                            "input": text  # 发送单个字符串
                        }
                        resp = client.post(self.api_url, json=payload, headers=headers)
                        if resp.status_code == 200:
                            emb = resp.json()['data'][0]['embedding']
                            results.append(emb)
                            break
                        elif resp.status_code == 429:
                            time.sleep(2)
                    except Exception:
                        time.sleep(1)
                else:
                    # 如果重试3次都失败，填入一个零向量占位，防止程序崩溃
                    print(f"第 {i + 1} 条数据向量化失败，跳过。")
                    results.append([0.0] * self.dimension)

        return results