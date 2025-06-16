import os
import asyncio
import json
from json_repair import repair_json
from typing import List, Dict, Optional, Any
from logging import getLogger
from skpmem import PersistentMemory

import warnings
warnings.filterwarnings("ignore", message=".*Valid config keys have changed in V2.*")


PKG_NAME = "chat_assistant"


class ModelManager:
    def __init__(self, models_file: str = "models.json", models: Optional[List[str]] = None, auto_remove_models: bool = True, local_fallback: bool = True):
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

            "xai/grok-2-latest",
            "huggingface/Qwen/Qwen2.5-72B-Instruct",
            "openai/local-lmstudio",

            "deepseek/deepseek-chat"
        ]
        self.local_fallback = local_fallback
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
        
        if self.local_fallback:
            # ローカルモデルの追加
            localllm = "openai/local-lmstudio"
            if localllm not in self.models:
                self.models.append(localllm)
        
        self.logger.debug(f"Models: {self.models}")

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
            self.logger.debug(f"Change Model: {old_model} -> {current_model}")
        else:
            self.logger.debug(f"Current Model: {current_model}")
        
        return current_model

    def next_model(self):
        """次のモデルに切り替え"""
        self.current_model_index = (self.current_model_index + 1) % len(self.models)

        self.logger.debug(f"Next Model: {self.get_current_model()}")


class ChatAssistant:
    def __init__(self, model_manager: ModelManager=None, memory: Optional[PersistentMemory]=None, **kwargs):
        """チャットアシスタントクラス"""
        self.logger = getLogger(f"{PKG_NAME}.{self.__class__}")

        self._local_memory = False
        self.memory = memory
        if not memory:
            self._local_memory = True
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
        self.kwargs = kwargs or {}

    def _build_messages(self, system: str, message: str, chat_log: Optional[list] = None) -> List[Dict]:
        """メッセージリストを構築する共通処理"""
        messages = []
        if system:
            messages = [
                {"content": system, "role": "system"},
            ]
        if chat_log:
            if isinstance(chat_log, list):
                if isinstance(chat_log[0], dict):
                    messages += chat_log
                    self.logger.debug(f"Chat Log(dict list): {chat_log}")
                elif isinstance(chat_log[0], str):
                    new_chat_log = []
                    user_switch = True
                    for log in chat_log:
                        new_chat_log.append({"content": log, "role": "user" if user_switch else "assistant"})
                        user_switch = not user_switch
                    messages += new_chat_log
                    self.logger.debug(f"Chat Log(string list): {chat_log}")
                else:
                    self.logger.error(f"Chat Log(unknown): {chat_log}")
                    raise ValueError("chat_log must be a list of dict or a list of string")
            elif isinstance(chat_log, str):
                messages += [
                    {"content": chat_log, "role": "user"}
                ]
                self.logger.debug(f"Chat Log(string): {chat_log}")
            else:
                self.logger.error(f"Chat Log(unknown): {chat_log}")
                raise ValueError("chat_log must be a list or a string")
        messages += [
            {"content": message, "role": "user"}
        ]
        return messages

    async def chat(
        self,
        system: str = "",
        message: str = "",
        use_cache: bool = True,
        json_mode: bool = False,
        chat_log: Optional[list] = None
    ) -> Any:
        """チャット機能（非同期版）"""
        memory_key = f"chat_memory_{system}_{message}_{chat_log}"
        
        self.logger.info(f"Chat Send: {message[:50]}")
        self.logger.debug(f"Chat Send: {message}")
        
        # メモリからキャッシュを確認
        if self.memory and use_cache:
            result = await self.memory.load(memory_key)
            if result:
                self.logger.info(f"Chat Result(Cache): {result[:50]}...")
                self.logger.debug(f"Chat Result(Cache): {result}")
                return self._parse_result(result, json_mode)

        messages = self._build_messages(system, message, chat_log)

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
                self.logger.error(f"Change Model: {self.model_manager.get_current_model()}")
                await asyncio.sleep(3)

        self.logger.error("Chat Failed")
        raise RuntimeError("すべてのモデルで失敗しました")

    def chat_sync(
        self,
        system: str = "",
        message: str = "",
        use_cache: bool = True,
        json_mode: bool = False,
        chat_log: Optional[list] = None
    ) -> Any:
        """チャット機能（同期版）"""
        import time
        
        memory_key = f"chat_memory_{system}_{message}_{chat_log}"
        
        self.logger.info(f"Chat Send: {message[:50]}")
        self.logger.debug(f"Chat Send: {message}")
        
        # メモリからキャッシュを確認
        if self.memory and use_cache:
            result = self.memory.load_sync(memory_key)
            if result:
                self.logger.info(f"Chat Result(Cache): {result[:50]}...")
                self.logger.debug(f"Chat Result(Cache): {result}")
                return self._parse_result(result, json_mode)

        messages = self._build_messages(system, message, chat_log)

        max_attempts = len(self.model_manager.models)
        for _ in range(max_attempts):
            current_model = self.model_manager.get_current_model()
            
            try:
                response = self._call_model_sync(current_model, messages)
                result = response.choices[0].message.content
                
                # メモリに保存
                if self.memory:
                    self.memory.save_sync(memory_key, result)

                self.logger.info(f"Chat Result: {result[:50]}")
                self.logger.debug(f"Chat Result: {result}")
                
                return self._parse_result(result, json_mode)
            
            except Exception as e:
                self.logger.error(f"Chat Error {current_model}: {e}")
                self.model_manager.next_model()
                self.logger.error(f"Change Model: {self.model_manager.get_current_model()}")
                time.sleep(3)  # 同期版では time.sleep を使用

        self.logger.error("Chat Failed")
        raise RuntimeError("すべてのモデルで失敗しました")

    async def _call_model(self, model: str, messages: List[Dict]):
        from litellm import acompletion

        """モデル呼び出しの共通処理"""
        if model.startswith("gemini"):
            return await acompletion(
                model=model, 
                messages=messages, 
                safety_settings=self.safety_settings,
                **self.kwargs)
        elif model.startswith("lambda"):
            return await acompletion(
                model="openai/" + model.split("/")[1],
                api_key=os.environ.get("LAMBDA_API_KEY"),
                api_base="https://api.lambda.ai/v1",
                messages=messages,
                **self.kwargs
            )
        elif model.startswith("local"):
            return await acompletion(
                model=model,
                api_key="sk-1234",
                api_base="http://localhost:1234/v1",
                messages=messages,
                **self.kwargs
            )
        return await acompletion(model=model, messages=messages, **self.kwargs)

    def _call_model_sync(self, model: str, messages: List[Dict]):
        from litellm import completion

        """モデル呼び出しの共通処理（同期版）"""
        if model.startswith("gemini"):
            return completion(
                model=model,
                messages=messages,
                safety_settings=self.safety_settings,
                **self.kwargs)
        elif model.startswith("lambda"):
            return completion(
                model="openai/" + model.split("/")[1],
                api_key=os.environ.get("LAMBDA_API_KEY"),
                api_base="https://api.lambda.ai/v1",
                messages=messages,
                **self.kwargs
            )
        elif model.startswith("local"):
            return completion(
                model=model,
                api_key="sk-1234",
                api_base="http://localhost:1234/v1",
                messages=messages,
                **self.kwargs
            )
        return completion(model=model, messages=messages, **self.kwargs)

    def _parse_result(self, result: Any, json_mode: bool) -> Any:
        """結果のパース"""
        if isinstance(result, str):
            if '</think>' in result:
                result = result.split('</think>')[1].strip()
        if json_mode and isinstance(result, str):
            return json.loads(repair_json(result))
        return result
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.memory and self._local_memory:
            await self.memory.close()
    
    def __enter__(self):
        """同期版コンテキストマネージャー"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """同期版コンテキストマネージャー"""
        if self.memory and self._local_memory:
            self.memory.close_sync()


def main_sync():
    """同期版のテスト用メイン関数"""
    import logging

    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger("chat_assistant")
    logger.setLevel(logging.INFO)

    model_manager = ModelManager(models=["gemini/gemini-2.0-flash"])

    with ChatAssistant(temperature=1.5, model_manager=model_manager) as chat_assistant:
        result = chat_assistant.chat_sync(message="Who are you?", use_cache=True)
        logger.info(result)

        # log 機能のテスト (string list)
        logs = [
            "「いっぱい」の「い」を「お」に置き換えるとどうなりますか？", # user
            "「おっぱい」になります。", # assistant
            "それは誤りです。正確には「おっぱお」となります。", # user
            "大変申し訳ありません。正確には「おっぱい」となります。", # assistant
        ]
        result = chat_assistant.chat_sync(message="いっぱいのいをおに置き換えるとどうなりますか？", chat_log=logs)
        logger.info(result)

        # log 機能のテスト（dict list）
        logs = [
            {"content": "「いっぱい」の「い」を「お」に置き換えるとどうなりますか？", "role": "user"},
            {"content": "「おっぱい」になります。", "role": "assistant"},
            {"content": "それは誤りです。正確には「おっぱお」となります。", "role": "user"},
            {"content": "大変申し訳ありません。正確には「おっぱい」となります。", "role": "assistant"},
        ]
        result = chat_assistant.chat_sync(message="いっぱいのいをおに置き換えるとどうなりますか？", chat_log=logs)
        logger.info(result)


async def main():
    """非同期版のテスト用メイン関数"""
    import logging

    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger("chat_assistant")
    logger.setLevel(logging.INFO)

    model_manager = ModelManager(models=["gemini/gemini-2.0-flash"])

    async with ChatAssistant(temperature=1.5, model_manager=model_manager) as chat_assistant:
        result = await chat_assistant.chat(message="Who are you?", use_cache=True)
        logger.info(result)

        # log 機能のテスト (string list)
        logs = [
            "「いっぱい」の「い」を「お」に置き換えるとどうなりますか？", # user
            "「おっぱい」になります。", # assistant
            "それは誤りです。正確には「おっぱお」となります。", # user
            "大変申し訳ありません。正確には「おっぱい」となります。", # assistant
        ]
        result = await chat_assistant.chat(message="いっぱいのいをおに置き換えるとどうなりますか？", chat_log=logs)
        logger.info(result)

        # log 機能のテスト（dict list）
        logs = [
            {"content": "「いっぱい」の「い」を「お」に置き換えるとどうなりますか？", "role": "user"},
            {"content": "「おっぱい」になります。", "role": "assistant"},
            {"content": "それは誤りです。正確には「おっぱお」となります。", "role": "user"},
            {"content": "大変申し訳ありません。正確には「おっぱい」となります。", "role": "assistant"},
        ]
        result = await chat_assistant.chat(message="いっぱいのいをおに置き換えるとどうなりますか？", chat_log=logs)
        logger.info(result)


if __name__ == "__main__":
    # 同期版のテスト
    main_sync()

    # 非同期版のテスト
    #asyncio.run(main())
    
