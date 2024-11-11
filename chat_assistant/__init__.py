import os
import asyncio
import json
from litellm import acompletion
from json_repair import repair_json
from typing import List, Dict, Optional, Any
from logging import getLogger
from pmem.async_pmem import PersistentMemory


PKG_NAME = "chat_assistant"


class ModelManager:
    def __init__(self, models_file: str = "models.json", models: Optional[List[str]] = None, auto_remove_models: bool = True):
        """モデル管理クラス"""
        self.logger = getLogger(f"{PKG_NAME}.{self.__class__}")
        self.models_file = models_file
        self.models = models or self._load_models() or [
            "gemini/gemini-1.5-pro-002",
            "openai/gpt-4o-2024-08-06",
            "anthropic/claude-3-5-sonnet-20241022",
            "cohere/command-r-plus-08-2024",

            "gemini/gemini-1.5-flash-002",
            "openai/gpt-4o-mini-2024-07-18",
            "anthropic/claude-3-5-haiku-20241022",
            "cohere/command-r-08-2024",

            "xai/grok-beta",
            "huggingface/Qwen/Qwen2.5-72B-Instruct",
            "openai/local-lmstudio",
        ]
        if auto_remove_models:
            self._remove_disable_models()
        self.current_model_index = 0
    
    def _remove_disable_models(self):
        """無効なAPIキーを持つモデルを削除"""

        # environment name = product name
        api_keys = {
            "OPENAI_API_KEY":"openai",
            "ANTHROPIC_API_KEY":"anthropic",
            "COHERE_API_KEY":"cohere",
            "GEMINI_API_KEY":"gemini",
            "XAI_API_KEY":"xai",
            "HUGGINGFACE_API_KEY":"huggingface"
        }

        # 環境変数が設定されていないモデルを削除
        for key, product_name in api_keys.items():
            if not os.environ.get(key):
                self.models = [model for model in self.models if product_name not in model]
        
        # ローカルモデルの追加
        localllm = "openai/local-lmstudio"
        if localllm not in self.models:
            self.models.append(localllm)
        
        self.logger.info(f"Models: {self.models}")

    def _load_models(self) -> List[str]:
        """モデルリストの読み込み"""
        if not os.path.exists(self.models_file):
            self.logger.debug(f"Models File Not Found: {self.models_file}")
            return None
        
        with open(self.models_file, "r") as f:
            self.logger.debug(f"Load Models: {self.models_file}")
            return json.load(f)

    def get_current_model(self) -> str:
        """現在のモデルを取得"""
        return self.models[self.current_model_index]

    def change_model(self, model_name: str) -> str:
        """指定されたモデルに変更"""

        old_model = self.get_current_model()

        try:
            self.current_model_index = self.models.index(model_name)
        except ValueError:
            # 部分一致で検索
            for i, model in enumerate(self.models):
                if model_name in model:
                    self.current_model_index = i
                    break
        
        current_model = self.get_current_model()
        
        if old_model != current_model:
            self.logger.info(f"Change Model: {old_model} -> {current_model}")
        else:
            self.logger.info(f"Current Model: {current_model}")
        
        return current_model

    def next_model(self):
        """次のモデルに切り替え"""
        self.current_model_index = (self.current_model_index + 1) % len(self.models)

        self.logger.info(f"Next Model: {self.get_current_model()}")


class ChatAssistant:
    def __init__(self, model_manager: ModelManager=None, memory: Optional[PersistentMemory]=None):
        """チャットアシスタントクラス"""
        self.logger = getLogger(f"{PKG_NAME}.{self.__class__}")

        self.memory = memory
        if not memory:
            self.memory = PersistentMemory("chat_log.db")
        if not model_manager:
            model_manager = ModelManager()
        self.model_manager = model_manager
        self.safety_settings = [
            {"category": category, "threshold": "BLOCK_NONE"}
            for category in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            ]
        ]

    async def chat(
        self, 
        system: str = "", 
        message: str = "", 
        use_cache: bool = True, 
        json_mode: bool = False
    ) -> Any:
        """チャット機能"""
        memory_key = f"chat_memory_{system}_{message}"
        
        # メモリからキャッシュを確認
        if self.memory and use_cache:
            result = await self.memory.load(memory_key)
            if result:
                self.logger.info(f"Chat Cache: {result[:50]}...")
                self.logger.debug(f"Chat Cache: {result}")
                return self._parse_result(result, json_mode)

        messages = [
            {"content": system, "role": "system"},
            {"content": message, "role": "user"}
        ]

        max_attempts = len(self.model_manager.models)
        for _ in range(max_attempts):
            current_model = self.model_manager.get_current_model()
            
            try:
                response = await self._call_model(current_model, messages)
                result = response.choices[0].message.content
                
                # メモリに保存
                if self.memory:
                    await self.memory.save(memory_key, result)

                self.logger.info(f"Chat Result: {result[:50]}")
                self.logger.debug(f"Chat Result: {result}")
                
                return self._parse_result(result, json_mode)
            
            except Exception as e:
                self.logger.error(f"Chat Error {current_model}: {e}")
                self.model_manager.next_model()
                await asyncio.sleep(3)

        self.logger.error("Chat Failed")
        raise RuntimeError("すべてのモデルで失敗しました")

    async def _call_model(self, model: str, messages: List[Dict]):
        """モデル呼び出しの共通処理"""
        if "gemini" in model:
            return await acompletion(
                model=model, 
                messages=messages, 
                safety_settings=self.safety_settings)
        elif "local" in model:
            return await acompletion(
                model=model,
                api_key="sk-1234",
                api_base="http://localhost:1234/v1",
                messages=messages
            )
        return await acompletion(model=model, messages=messages)

    def _parse_result(self, result: Any, json_mode: bool) -> Any:
        """結果のパース"""
        if json_mode and isinstance(result, str):
            return json.loads(repair_json(result))
        return result


async def main():
    # log output console
    import logging

    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # コンソールハンドラーを追加
        ]
    )
    
    logger = logging.getLogger("chat_assistant")
    logger.setLevel(logging.DEBUG)

    chat_assistant = ChatAssistant()
    chat_assistant.model_manager.change_model("xai")
    result = await chat_assistant.chat(message="Who are you???", use_cache=False)

    logger.info(result)

    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())