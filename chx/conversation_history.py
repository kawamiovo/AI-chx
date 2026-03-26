import asyncio
from typing import List
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from datetime import datetime
import uuid

from conversation import APIEmbeddingFunction
from embedding import EmbeddingService
from config import Config
import pandas as pd
import os


class ConversationTurn:#封装单轮对话的问答对
    def __init__(self, ask: str, answer: str):
        self.ask = ask
        self.answer = answer

    def __str__(self):
        return f"user: {self.ask}\nassistant: {self.answer}"


class ConversationHistory:  # 对话记忆保存到 save/memory 目录
    def __init__(self, max_turns: int = 20):
        self.turns = []
        self.max_turns = max_turns

        # 初始化向量数据库客户端
        self.client = chromadb.Client(Settings(
            persist_directory="save/chat_history",
            is_persistent=True
        ))

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(  # 创建数据集,collection类似关系型数据库的表
            name="history",
            embedding_function=APIEmbeddingFunction()
        )
        self._load_recent_history()

    def _load_recent_history(self):
        """从向量数据库取回最后 20 条，恢复对话状态"""
        try:
            # 获取数据库中所有的 id
            results = self.collection.get()
            if results['ids']:
                # 获取最后 20 条数据的索引
                # 注意：ChromaDB 默认不保证顺序，但我们后面按时间戳排一下
                count = len(results['ids'])
                start_idx = max(0, count - self.max_turns)

                # 获取最后的数据
                last_ids = results['ids'][start_idx:]
                last_docs = results['documents'][start_idx:]

                # 重新填充到 self.turns 列表里
                for doc in last_docs:
                    if "user: " in doc and "\nassistant: " in doc:
                        # 简单的解析逻辑
                        parts = doc.split("\nassistant: ")
                        ask = parts[0].replace("user: ", "")
                        answer = parts[1]
                        self.turns.append(ConversationTurn(ask, answer))
        except Exception as e:
            print(f"初始化加载历史记录失败: {e}")

    def add_dialog(self, user_message: str, assistant_message: str):
        """添加新对话，并立即保存到向量数据库"""
        turn = ConversationTurn(user_message, assistant_message)
        self.turns.append(turn)
        
        # 立即保存到向量数据库
        content = str(turn)
        self.collection.add(
            documents=[content],
            metadatas=[{
                "timestamp": datetime.now().isoformat()
            }],
            ids=[str(uuid.uuid4())]
        )

        # 当对话数量达到最大值时，移除最早的对话
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_history_list(self):
        return [{"ask": t.ask, "answer": t.answer} for t in self.turns]

    def _auto_archive(self):
        """自动归档一半的对话"""
        if not self.turns:
            return

        # 计算要归档的对话数量
        archive_count = len(self.turns) // 2

        # 准备归档内容
        archive_turns = self.turns[:archive_count]
        content = "\n".join(str(turn) for turn in archive_turns)

        print("以下内容将被归档：")
        print(content)
        print("--------------------------------")

        # 保存到向量数据库
        self.collection.add(
            documents=[content],
            metadatas=[{
                "timestamp": datetime.now().isoformat()
            }],
            ids=[str(uuid.uuid4())]
        )

        # 移除已归档的对话
        self.turns = self.turns[archive_count:]

    def get_context(self) -> str:
        """获取格式化后的对话上下文"""
        return "\n".join(str(turn) for turn in self.turns)

    def retrieve(self, user_message: str, n_results: int = 3) -> List[str]:
        """获取与用户消息最相关的历史记忆"""
        results = self.collection.query(  # 基于用户当前消息查找相关历史对话
            query_texts=[user_message],
            n_results=n_results,
            include=['documents']
        )

        return results['documents'][0] if results['documents'] else []
