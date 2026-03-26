import json
import os
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import chromadb
from config import Config
from llm import LLMService
from conversation_history import ConversationHistory
from memory_core import AsyncEmbeddingGenerator  # 确保从 memory_core 导入


class MainAgent:
    def __init__(self, llm_service: LLMService, conversation_history: ConversationHistory):
        self.llm_service = llm_service
        self.conversation_history = conversation_history

        # 1. 初始化向量数据库客户端 (路径必须与 memory_core.py 保持完全一致)
        self.client = chromadb.PersistentClient(path="./save/memory_advanced")

        # 2. 获取或创建集合 (避免集合不存在时崩溃)
        self.collection = self.client.get_or_create_collection(
            name="ex_gf_contextual_memory"
        )

        # 3. 初始化异步向量生成器 (用于搜索时转换用户问题)
        self.embedder = AsyncEmbeddingGenerator()

        # 4. 读取回复模板 (使用绝对路径增加稳定性)
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(self.base_path, 'prompts', 'reply.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                self.prompt_template = file.read()
        except FileNotFoundError:
            print(f"警告: 找不到模板文件 {prompt_path}，将使用默认模板")
            self.prompt_template = "你现在扮演前女友。对话记录：{chat_history}\n记忆：{memory}\n用户说：{user_message}"

        # 5. 确保日志目录存在
        self.log_dir = os.path.join(self.base_path, 'save', 'log')
        os.makedirs(self.log_dir, exist_ok=True)

        # 6. 读取个人信息
        self.user_info_file = os.path.join(self.base_path, 'save', 'me.txt')
        self.user_info = self._load_user_info()

    def _load_user_info(self) -> str:
        """加载用户个人信息"""
        if os.path.exists(self.user_info_file):
            try:
                with open(self.user_info_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"读取个人信息出错: {e}")
        return "用户没有提供特别的个人信息。"

    def _log_conversation(self, role: str, content: str) -> None:
        """记录对话到日志文件"""
        current_date = datetime.now().strftime('%Y%m%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        log_file = os.path.join(self.log_dir, f'{current_date}.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'[{current_time}] {role}: {content}\n')

    async def _get_relevant_memories(self, message: str) -> str:
        """核心逻辑：从向量数据库检索相关记忆"""
        try:
            # 将当前用户输入转化为向量
            # 使用列表包装以适配批量接口 [message]
            query_vectors = await self.embedder.generate_batch([message])

            if not query_vectors or not query_vectors[0]:
                return "（未找到相关历史记忆）"

            # 在 ChromaDB 中进行向量搜索
            results = self.collection.query(
                query_embeddings=[query_vectors[0]],
                n_results=5,  # 返回最相关的5条记忆
                include=['documents']
            )

            if results['documents'] and results['documents'][0]:
                relevant_texts = results['documents'][0]
                return "\n---\n".join(relevant_texts)

            return "（无相关记忆）"

        except Exception as e:
            print(f"搜索记忆库出错: {e}")
            return "（记忆系统暂时离线）"

    async def _generate_reply(self, message: str, memory_text: str) -> Tuple[str, str]:
        """调用 LLM 生成对话回复"""
        # 获取多轮对话上下文
        current_context = self.conversation_history.get_context()

        try:
            # 填充 Prompt 模板
            full_prompt = self.prompt_template.format(
                chat_history=current_context,
                user_message=message,
                memory=memory_text,
                user_info=self.user_info
            )

            # 调用 LLM 服务
            raw_response = await self.llm_service.generate_response(full_prompt)
            if not raw_response:
                return "你说什么？我刚才走神了。", "委屈"

            # 清洗 LLM 返回的内容，防止它返回 ```json ... ``` 这种 Markdown 格式
            clean_response = raw_response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()

            # 解析 JSON 回复
            try:
                data = json.loads(clean_response)
                reply = data.get("reply", "...")
                expression = data.get("expression", "开心")
                return reply, expression
            except json.JSONDecodeError:
                # 如果 LLM 没有按要求返回 JSON，则直接返回纯文本
                print("警告: LLM 返回非标准 JSON 格式，执行降级处理")
                return clean_response, "开心"

        except Exception as e:
            print(f"生成回复逻辑出错: {e}")
            return "（盒子陷入了沉思...）", "难过"

    async def reply(self, message: str) -> Tuple[str, str]:
        """主入口：处理输入并输出回复"""
        # 1. 记录日志
        self._log_conversation('User', message)

        # 2. 检索历史记忆 (语义搜索)
        memory_text = await self._get_relevant_memories(message)
        print(f"--- 检索到的相关记忆 ---\n{memory_text}\n-----------------------")

        # 3. 生成 AI 回复
        reply_content, expression = await self._generate_reply(message, memory_text)

        # 4. 记录日志及对话历史
        self._log_conversation('Assistant', f"[{expression}] {reply_content}")
        self.conversation_history.add_dialog(message, reply_content)

        return reply_content, expression


# ==========================================
# 本地测试代码 (直接运行 main_agent.py 可测试)
# ==========================================
if __name__ == "__main__":
    async def test():
        # 这里需要导入你其他的服务类进行测试
        # 仅作演示
        print("正在启动测试...")
        # agent = MainAgent(llm_service, ConversationHistory())
        # reply, expr = await agent.reply("你还记得我们第一次见面吗？")
        # print(f"回复: {reply}, 表情: {expr}")


    asyncio.run(test())