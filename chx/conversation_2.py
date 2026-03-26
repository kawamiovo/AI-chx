import asyncio
import pickle
import math
import os
import pandas as pd
import jieba
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from config import Config

# 假设你的 stopwords 放在这里
STOPWORDS_DIR = r"./stopwords"


@dataclass
class SearchResult:
    content: str  # 匹配到的核心内容
    context_pair: str  # 上下文对话（问答对），用于给AI参考语气
    score: float  # 匹配分数
    time_str: str  # 时间
    is_sender: int  # 是谁说的 (1是本人, 0是前女友)


class TextAnalyzer:
    def __init__(self, stopwords_dir: str = STOPWORDS_DIR):
        self.stopwords = self._load_stopwords(stopwords_dir)
        self.df = None
        self.corpus_tokens = []  # 存储分词后的语料
        self.doc_lengths = []  # 记录每条记录的长度
        self.avgdl = 0  # 平均文档长度
        self.idf = {}  # 逆文档频率

    def _load_stopwords(self, stopwords_dir: str) -> set:
        """加载停用词，增加了容错处理"""
        all_stopwords = set()
        if not os.path.exists(stopwords_dir):
            print(f"提示: 停用词目录 {stopwords_dir} 不存在，将仅使用jieba默认过滤")
            return all_stopwords

        for filename in os.listdir(stopwords_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(stopwords_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        all_stopwords.update(line.strip() for line in f)
                except Exception as e:
                    print(f"加载 {filename} 失败: {e}")
        return all_stopwords

    def load_data(self, excel_path: str) -> None:
        """
        加载数据并预处理。
        注意：这里假设 CSV 有 'IsSender' 列 (1=我, 0=她) 和 'StrTime' 列。
        如果没有，请根据实际表头修改。
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"文件未找到: {excel_path}")

        self.df = pd.read_csv(excel_path)

        # 填充缺失值
        required_cols = ['StrContent', 'IsSender', 'StrTime']
        for col in required_cols:
            if col not in self.df.columns:
                # 如果没有 IsSender，默认全当成前女友的记录（但这会降低效果）
                if col == 'IsSender':
                    self.df['IsSender'] = 0
                elif col == 'StrTime':
                    self.df['StrTime'] = ''
                else:
                    self.df[col] = ''

        self.df['StrContent'] = self.df['StrContent'].astype(str).fillna("")

    def build_index(self, cache_file: str = "corpus_index.pkl") -> None:
        """
        构建 BM25 索引（一种比单纯关键词匹配更准的算法）
        """
        if os.path.exists(cache_file):
            print("加载缓存的索引...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.corpus_tokens = data['tokens']
                self.idf = data['idf']
                self.avgdl = data['avgdl']
            return

        print("正在构建索引（分词中）...")
        # 1. 分词
        texts = self.df['StrContent'].tolist()
        self.corpus_tokens = []
        for text in texts:
            words = [w for w in jieba.lcut(text) if w not in self.stopwords and len(w.strip()) > 0]
            self.corpus_tokens.append(words)

        # 2. 计算 BM25 所需的统计量
        doc_count = len(self.corpus_tokens)
        self.doc_lengths = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / doc_count if doc_count > 0 else 0

        # 计算 IDF
        word_doc_freq = {}
        for doc in self.corpus_tokens:
            for word in set(doc):  # set去重，一句话里出现两次算一次文档频率
                word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

        self.idf = {}
        for word, freq in word_doc_freq.items():
            # BM25 的 IDF 公式
            self.idf[word] = math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)

        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'tokens': self.corpus_tokens,
                'idf': self.idf,
                'avgdl': self.avgdl
            }, f)
        print("索引构建完成。")

    def _bm25_score(self, query_words: List[str], doc_index: int, k1=1.5, b=0.75) -> float:
        """计算单条记录的 BM25 分数"""
        score = 0.0
        doc_tokens = self.corpus_tokens[doc_index]
        doc_len = len(doc_tokens)

        if doc_len == 0: return 0.0

        # 统计文档中词频
        doc_freqs = {}
        for w in doc_tokens:
            doc_freqs[w] = doc_freqs.get(w, 0) + 1

        for word in query_words:
            if word not in doc_tokens:
                continue

            idf = self.idf.get(word, 0)
            tf = doc_freqs[word]

            # BM25 核心公式
            numerator = idf * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / self.avgdl))
            score += numerator / denominator

        return score

    async def search(self, query: str, k: int = 3, mode: str = 'hybrid') -> List[SearchResult]:
        """
        搜索函数
        :param query: 用户当前的问题
        :param k: 返回条数
        :param mode:
            - 'memory': 搜索前女友说过的话（用于提取事实记忆）
            - 'mimic': 搜索【我】说过的话，并返回【她】的回复（用于模仿语气）
            - 'hybrid': 混合模式（推荐）
        """
        if not self.corpus_tokens:
            raise ValueError("索引未构建")

        query_words = [w for w in jieba.lcut(query) if w not in self.stopwords]

        scores = []
        for idx, _ in enumerate(self.corpus_tokens):
            score = self._bm25_score(query_words, idx)
            if score > 0:
                scores.append((idx, score))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_indices = set()  # 去重

        for idx, score in scores:
            if len(results) >= k:
                break

            if idx in seen_indices:
                continue

            row = self.df.iloc[idx]
            current_is_sender = row['IsSender']  # 1=我, 0=她

            # 逻辑处理：
            # 如果是 mode='mimic' (模仿语气)，我们要找【我】说过类似的话(idx)，然后取【她】的下一句(idx+1)作为参考
            final_content = ""
            context_pair = ""

            # 边界检查
            next_idx = idx + 1
            has_next = next_idx < len(self.df)

            if mode == 'mimic':
                # 只关心“我”说过的类似的话，且下一句是“她”说的
                if current_is_sender == 1 and has_next and self.df.iloc[next_idx]['IsSender'] == 0:
                    my_past_question = row['StrContent']
                    her_past_reply = self.df.iloc[next_idx]['StrContent']
                    final_content = her_past_reply
                    context_pair = f"用户曾问：{my_past_question} -> 前女友回复：{her_past_reply}"
                    # 分数稍微降低，因为是间接匹配，但相关性其实很高
                    score *= 1.2
                else:
                    continue  # 不符合模仿逻辑，跳过

            elif mode == 'memory':
                # 只关心“她”说过的话
                if current_is_sender == 0:
                    final_content = row['StrContent']
                    context_pair = f"前女友曾说：{final_content}"
                else:
                    continue

            else:  # hybrid
                # 混合模式：优先找回复对，其次找直接陈述
                if current_is_sender == 1 and has_next and self.df.iloc[next_idx]['IsSender'] == 0:
                    # 找到了以前的问答对，这是极好的语气参考
                    context_pair = f"历史对话参考(User: {row['StrContent']} -> GF: {self.df.iloc[next_idx]['StrContent']})"
                    final_content = context_pair
                elif current_is_sender == 0:
                    # 找到了以前她的陈述
                    context_pair = f"前女友记忆碎片: {row['StrContent']}"
                    final_content = context_pair
                else:
                    continue

            results.append(SearchResult(
                content=final_content,
                context_pair=context_pair,
                score=score,
                time_str=str(row['StrTime']),
                is_sender=current_is_sender
            ))
            seen_indices.add(idx)

        return results


async def main():
    analyzer = TextAnalyzer()
    excel_path = r"E:\pythonlearn\逆向\前女友\data\merged_output.csv"

    # 1. 加载数据
    try:
        analyzer.load_data(excel_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 构建或加载索引
    analyzer.build_index()

    # 3. 模拟虚拟恋人场景
    query = "上班咋样"
    print(f"\n当前用户输入: {query}")

    # 搜索：这里我们使用 mimic 模式，试图找到以前你是怎么问的，她是怎么回的
    print("-" * 30)
    print("【尝试寻找语气参考 (Mimic Mode)】")
    results_mimic = await analyzer.search(query, k=2, mode='mimic')
    for res in results_mimic:
        print(f"匹配度: {res.score:.2f} | 时间: {res.time_str}")
        print(f"参考条目: {res.context_pair}")

    print("-" * 30)
    print("【尝试寻找事实记忆 (Memory Mode)】")
    results_mem = await analyzer.search(query, k=2, mode='memory')
    for res in results_mem:
        print(f"匹配度: {res.score:.2f} | 时间: {res.time_str}")
        print(f"记忆内容: {res.content}")


if __name__ == "__main__":
    asyncio.run(main())