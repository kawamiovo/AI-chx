import httpx
import asyncio
from typing import List, Dict
import json
import re
from openai import OpenAI

class LLMService:
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        
    async def generate_response(
        self, 
        message: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        is_json: bool = False
    ) -> str:
        retry_count = 0
        print("主人的消息是：",message)
        # while retry_count < max_retries:
        try:
            client = OpenAI(api_key=self.api_key,
                            base_url=self.api_url)
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": message},
                ],
                stream=False
            )
            raw_response = response.choices[0].message.content.strip()
            print("原生回复：",raw_response)
            return raw_response

        except Exception as e:
            retry_count += 1
            print(f"LLM Error (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                await asyncio.sleep(1)
            return "对不起，我遇到了一些问题，请稍后再试。"


