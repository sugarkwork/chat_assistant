import os
import asyncio
import json
import threading
import hashlib
from json_repair import loads
from typing import List, Dict, Optional, Any, Union
from logging import getLogger
from skpmem import PersistentMemory
import yaml
from dotenv import load_dotenv
load_dotenv()

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
        self._lock = threading.Lock()  # スレッドセーフティのためのロック
    
    def _remove_disable_models(self):
        """無効なAPIキーを持つモデルを削除"""

        # environment name = product name
        api_keys = {
            "OPENAI_API_KEY":"openai",
            "ANTHROPIC_API_KEY":"anthropic",
            "COHERE_API_KEY":"cohere",
            "GEMINI_API_KEY":"gemini",
            "XAI_API_KEY":"xai",
            "HUGGINGFACE_API_KEY":"huggingface",
            "DEEPSEEK_API_KEY":"deepseek"
        }

        removed_models = []
        # 環境変数が設定されていないモデルを削除
        for key, product_name in api_keys.items():
            if not os.environ.get(key):
                # 削除対象のモデルを記録
                models_to_remove = [model for model in self.models if product_name in model]
                if models_to_remove:
                    removed_models.extend([(model, key) for model in models_to_remove])
                    self.models = [model for model in self.models if product_name not in model]
        
        # 削除されたモデルの警告を表示
        if removed_models:
            self.logger.warning("以下のモデルは対応するAPIキーが設定されていないため削除されました:")
            for model, key in removed_models:
                self.logger.warning(f"  - {model} (環境変数 {key} を設定してください)")
            
            # APIキー設定例を表示
            self.logger.info("APIキーの設定例:")
            unique_keys = set(key for _, key in removed_models)
            for key in unique_keys:
                self.logger.info(f"  export {key}=\"your-api-key-here\"")
        
        if self.local_fallback:
            # ローカルモデルの追加
            localllm = "openai/local-lmstudio"
            if localllm not in self.models:
                self.models.append(localllm)
                self.logger.info(f"ローカルフォールバックモデルを追加しました: {localllm}")
                self.logger.info("  注: ローカルモデルを使用するには、LM Studioなどが http://localhost:1234 で実行されている必要があります")
        
        if not self.models:
            self.logger.error("利用可能なモデルがありません！以下のいずれかを設定してください:")
            for key, product_name in api_keys.items():
                self.logger.error(f"  - {key} ({product_name}モデル用)")
            self.logger.error("  または、LM Studioなどのローカルサーバーを起動してください")
        
        self.logger.debug(f"最終的なモデルリスト: {self.models}")

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
        with self._lock:
            if not self.models:
                raise ValueError("No models available")
            return self.models[self.current_model_index]

    def change_model(self, model_name: str) -> str:
        """指定されたモデルに変更"""
        with self._lock:
            old_model = self.models[self.current_model_index] if self.models else None

            try:
                self.current_model_index = self.models.index(model_name)
            except ValueError:
                # 部分一致で検索
                found = False
                for i, model in enumerate(self.models):
                    if model_name in model:
                        self.current_model_index = i
                        found = True
                        break
                
                if not found:
                    # 見つからない場合は例外を発生
                    raise ValueError(f"Model '{model_name}' not found in available models: {self.models}")
            
            current_model = self.models[self.current_model_index]
            
            if old_model != current_model:
                self.logger.debug(f"Change Model: {old_model} -> {current_model}")
            else:
                self.logger.debug(f"Current Model: {current_model}")
            
            return current_model

    def next_model(self):
        """次のモデルに切り替え"""
        with self._lock:
            if not self.models:
                raise ValueError("No models available")
            self.current_model_index = (self.current_model_index + 1) % len(self.models)
            current_model = self.models[self.current_model_index]
            
        self.logger.debug(f"Next Model: {current_model}")


class ChatAssistant:
    def __init__(self, model_manager: ModelManager=None, memory: Optional[PersistentMemory]=None, retry_sleep: float = 3.0, **kwargs):
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
        self.retry_sleep = retry_sleep  # 再試行時のスリープ時間
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

    def _generate_memory_key(self, system: str, message: str, chat_log: Optional[Union[List[Dict], List[str], str]] = None) -> str:
        """効率的なメモリキーを生成"""
        # 大きなデータをハッシュ化してキーサイズを削減
        data_str = f"{system}_{message}_{chat_log}"
        if len(data_str) > 200:  # 長いデータはハッシュ化
            return f"chat_memory_{hashlib.md5(data_str.encode()).hexdigest()}"
        return f"chat_memory_{data_str}"

    def _build_messages(self, system: str, message: str, chat_log: Optional[Union[List[Dict], List[str], str]] = None) -> List[Dict[str, str]]:
        """メッセージリストを構築する共通処理"""
        messages = []
        if system:
            messages = [
                {"content": system, "role": "system"},
            ]
        if chat_log:
            if isinstance(chat_log, list):
                if len(chat_log) == 0:
                    # 空リストの場合は何もしない
                    pass
                elif isinstance(chat_log[0], dict):
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
        chat_log: Optional[Union[List[Dict], List[str], str]] = None
    ) -> Any:
        """チャット機能（非同期版）"""
        memory_key = self._generate_memory_key(system, message, chat_log)
        
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
                
                # choicesの境界チェック
                if not response.choices or len(response.choices) == 0:
                    raise ValueError("No choices returned from model")
                
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
                await asyncio.sleep(self.retry_sleep)

        self.logger.error("Chat Failed")
        raise RuntimeError("すべてのモデルで失敗しました")

    def chat_sync(
        self,
        system: str = "",
        message: str = "",
        use_cache: bool = True,
        json_mode: bool = False,
        chat_log: Optional[Union[List[Dict], List[str], str]] = None
    ) -> Any:
        """チャット機能（同期版）"""
        import time
        
        memory_key = self._generate_memory_key(system, message, chat_log)
        
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
                
                # choicesの境界チェック
                if not response.choices or len(response.choices) == 0:
                    raise ValueError("No choices returned from model")
                
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
                time.sleep(self.retry_sleep)  # 同期版では time.sleep を使用

        self.logger.error("Chat Failed")
        raise RuntimeError("すべてのモデルで失敗しました")

    async def _call_model(self, model: str, messages: List[Dict]) -> Any:
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
        elif model.startswith("openai/local") or model.startswith("local"):
            return await acompletion(
                model=model,
                api_key="sk-1234",
                api_base="http://localhost:1234/v1",
                messages=messages,
                **self.kwargs
            )
        return await acompletion(model=model, messages=messages, **self.kwargs)

    def _call_model_sync(self, model: str, messages: List[Dict]) -> Any:
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
        elif model.startswith("openai/local") or model.startswith("local"):
            return completion(
                model=model,
                api_key="sk-1234",
                api_base="http://localhost:1234/v1",
                messages=messages,
                **self.kwargs
            )
        return completion(model=model, messages=messages, **self.kwargs)

    async def _call_model_stream(self, model: str, messages: List[Dict]) -> Any:
        from litellm import acompletion

        """ストリーミング版モデル呼び出しの共通処理（非同期）"""
        kwargs = {**self.kwargs, "stream": True}
        
        if model.startswith("gemini"):
            return await acompletion(
                model=model, 
                messages=messages, 
                safety_settings=self.safety_settings,
                **kwargs)
        elif model.startswith("lambda"):
            return await acompletion(
                model="openai/" + model.split("/")[1],
                api_key=os.environ.get("LAMBDA_API_KEY"),
                api_base="https://api.lambda.ai/v1",
                messages=messages,
                **kwargs
            )
        elif model.startswith("openai/local") or model.startswith("local"):
            return await acompletion(
                model=model,
                api_key="sk-1234",
                api_base="http://localhost:1234/v1",
                messages=messages,
                **kwargs
            )
        return await acompletion(model=model, messages=messages, **kwargs)

    def _call_model_stream_sync(self, model: str, messages: List[Dict]) -> Any:
        from litellm import completion

        """ストリーミング版モデル呼び出しの共通処理（同期）"""
        kwargs = {**self.kwargs, "stream": True}
        
        if model.startswith("gemini"):
            return completion(
                model=model,
                messages=messages,
                safety_settings=self.safety_settings,
                **kwargs)
        elif model.startswith("lambda"):
            return completion(
                model="openai/" + model.split("/")[1],
                api_key=os.environ.get("LAMBDA_API_KEY"),
                api_base="https://api.lambda.ai/v1",
                messages=messages,
                **kwargs
            )
        elif model.startswith("openai/local") or model.startswith("local"):
            return completion(
                model=model,
                api_key="sk-1234",
                api_base="http://localhost:1234/v1",
                messages=messages,
                **kwargs
            )
        return completion(model=model, messages=messages, **kwargs)

    def _parse_result(self, result: Any, json_mode: bool) -> Any:
        """結果のパース"""
        if isinstance(result, str):
            if '</think>' in result:
                result = result.split('</think>')[1].strip()
        if json_mode and isinstance(result, str):
            return loads(result)
        return result
    
    async def chat_stream(
        self,
        system: str = "",
        message: str = "",
        chat_log: Optional[Union[List[Dict], List[str], str]] = None
    ):
        """チャット機能（非同期ストリーミング版）"""
        self.logger.info(f"Chat Stream Send: {message[:50]}")
        self.logger.debug(f"Chat Stream Send: {message}")
        
        messages = self._build_messages(system, message, chat_log)
        
        max_attempts = len(self.model_manager.models)
        for _ in range(max_attempts):
            current_model = self.model_manager.get_current_model()
            
            try:
                response = await self._call_model_stream(current_model, messages)
                
                full_response = ""
                async for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield content
                        
                        if chunk.choices[0].finish_reason:
                            break
                
                self.logger.info(f"Chat Stream Result: {full_response[:50]}")
                self.logger.debug(f"Chat Stream Result: {full_response}")
                
                # ストリーミング成功時は終了
                return
                
            except Exception as e:
                self.logger.error(f"Chat Stream Error {current_model}: {e}")
                self.model_manager.next_model()
                self.logger.error(f"Change Model: {self.model_manager.get_current_model()}")
                await asyncio.sleep(self.retry_sleep)
        
        self.logger.error("Chat Stream Failed")
        raise RuntimeError("すべてのモデルでストリーミングに失敗しました")
    
    def chat_stream_sync(
        self,
        system: str = "",
        message: str = "",
        chat_log: Optional[Union[List[Dict], List[str], str]] = None
    ):
        """チャット機能（同期ストリーミング版）"""
        import time
        
        self.logger.info(f"Chat Stream Send: {message[:50]}")
        self.logger.debug(f"Chat Stream Send: {message}")
        
        messages = self._build_messages(system, message, chat_log)
        
        max_attempts = len(self.model_manager.models)
        for _ in range(max_attempts):
            current_model = self.model_manager.get_current_model()
            
            try:
                response = self._call_model_stream_sync(current_model, messages)
                
                full_response = ""
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield content
                        
                        if chunk.choices[0].finish_reason:
                            break
                
                self.logger.info(f"Chat Stream Result: {full_response[:50]}")
                self.logger.debug(f"Chat Stream Result: {full_response}")
                
                # ストリーミング成功時は終了
                return
                
            except Exception as e:
                self.logger.error(f"Chat Stream Error {current_model}: {e}")
                self.model_manager.next_model()
                self.logger.error(f"Change Model: {self.model_manager.get_current_model()}")
                time.sleep(self.retry_sleep)
        
        self.logger.error("Chat Stream Failed")
        raise RuntimeError("すべてのモデルでストリーミングに失敗しました")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        """非同期版コンテキストマネージャー終了処理"""
        if self.memory and self._local_memory:
            try:
                await self.memory.close()
            except Exception as e:
                self.logger.error(f"Error closing memory: {e}")
                # 例外を発生させずに処理を続行
        # False を返すことで、元の例外を再発生させる
        return False
    
    def __enter__(self):
        """同期版コンテキストマネージャー"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """同期版コンテキストマネージャー終了処理"""
        if self.memory and self._local_memory:
            try:
                self.memory.close_sync()
            except Exception as e:
                self.logger.error(f"Error closing memory: {e}")
                # 例外を発生させずに処理を続行
        # False を返すことで、元の例外を再発生させる
        return False


def main_sync():
    """同期版のテスト用メイン関数"""
    import logging

    logging.basicConfig(
        level=logging.INFO,  # 基本レベルをINFOに設定
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
            "APIの応答時間が遅いのですが、最適化方法はありますか？", # user
            "応答時間の最適化には複数のアプローチがあります。キャッシュの活用、バッチ処理、接続プールの設定などが効果的です。", # assistant
            "具体的なキャッシュ戦略について教えてください。", # user
            "Redis やメモリキャッシュを使用し、頻繁にアクセスされるデータを事前に読み込むことで高速化できます。", # assistant
        ]
        result = chat_assistant.chat_sync(message="キャッシュの実装例を教えてください。", chat_log=logs)
        logger.info(result)

        # log 機能のテスト（dict list）
        logs = [
            {"content": "APIの応答時間が遅いのですが、最適化方法はありますか？", "role": "user"},
            {"content": "応答時間の最適化には複数のアプローチがあります。キャッシュの活用、バッチ処理、接続プールの設定などが効果的です。", "role": "assistant"},
            {"content": "具体的なキャッシュ戦略について教えてください。", "role": "user"},
            {"content": "Redis やメモリキャッシュを使用し、頻繁にアクセスされるデータを事前に読み込むことで高速化できます。", "role": "assistant"},
        ]
        result = chat_assistant.chat_sync(message="キャッシュの実装例を教えてください。", chat_log=logs)
        logger.info(result)


async def main():
    """非同期版のテスト用メイン関数"""
    import logging

    logging.basicConfig(
        level=logging.INFO,  # 基本レベルをINFOに設定
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger("chat_assistant")
    logger.setLevel(logging.INFO)

    model_manager = ModelManager(models=["gemini/gemini-2.5-flash"])

    async with ChatAssistant(temperature=1.5, model_manager=model_manager) as chat_assistant:
        result = await chat_assistant.chat(message="Who are you?", use_cache=True)
        logger.info(result)

        # log 機能のテスト (string list)
        logs = [
            "APIの応答時間が遅いのですが、最適化方法はありますか？", # user
            "応答時間の最適化には複数のアプローチがあります。キャッシュの活用、バッチ処理、接続プールの設定などが効果的です。", # assistant
            "具体的なキャッシュ戦略について教えてください。", # user
            "Redis やメモリキャッシュを使用し、頻繁にアクセスされるデータを事前に読み込むことで高速化できます。", # assistant
        ]
        result = await chat_assistant.chat(message="キャッシュの実装例を教えてください。", chat_log=logs)
        logger.info(result)

        # log 機能のテスト（dict list）
        logs = [
            {"content": "APIの応答時間が遅いのですが、最適化方法はありますか？", "role": "user"},
            {"content": "応答時間の最適化には複数のアプローチがあります。キャッシュの活用、バッチ処理、接続プールの設定などが効果的です。", "role": "assistant"},
            {"content": "具体的なキャッシュ戦略について教えてください。", "role": "user"},
            {"content": "Redis やメモリキャッシュを使用し、頻繁にアクセスされるデータを事前に読み込むことで高速化できます。", "role": "assistant"},
        ]
        result = await chat_assistant.chat(message="キャッシュの実装例を教えてください。", chat_log=logs)
        logger.info(result)


def stream_demo_sync():
    """同期版ストリーミングのデモ"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger("chat_assistant")
    logger.setLevel(logging.INFO)

    models = ["lambda/deepseek-r1-0528", "gemini/gemini-2.5-flash", "lambda/deepseek-llama3.3-70b"]
    
    model_manager = ModelManager(models=models)

    with ChatAssistant(temperature=0.7, model_manager=model_manager) as chat_assistant:
        print(f"使用モデル: {model_manager.get_current_model()}")
        print("ストリーミング開始:")
        
        try:
            for chunk in chat_assistant.chat_stream_sync(message="Pythonの歴史について3文で説明してください。"):
                print(chunk, end='', flush=True)
            print("\n完了!")
        except Exception as e:
            print(f"\nエラー: {e}")
            print("\n注意: ローカルモデルを使用する場合は、LM Studioなどのローカルサーバーが http://localhost:1234 で実行されている必要があります。")
            print("APIキーを設定する場合は、環境変数を設定してください:")
            print("  - GEMINI_API_KEY")
            print("  - OPENAI_API_KEY")
            print("  - ANTHROPIC_API_KEY")
        
        # チャットログを使用したストリーミング
        chat_log = [
            "プログラミング言語の設計について教えてください。",
            "プログラミング言語の設計には、構文、型システム、実行モデルなど多くの要素が含まれます。"
        ]
        print("\nコンテキスト付きストリーミング:")
        for chunk in chat_assistant.chat_stream_sync(
            message="Pythonの設計思想は？", 
            chat_log=chat_log
        ):
            print(chunk, end='', flush=True)
        print("\n完了!")


async def chat_stream_data_process(chunk: str, format: str = "json") -> dict:
    think_support = False
    thinking = False
    body_start = False

    think_start_tag = "<think>"
    think_end_tag = "</think>"

    chunk = str(chunk).strip()

    if not chunk:
        return {}
    
    if len(chunk) < len(think_start_tag):
        return {}

    if chunk.startswith(think_start_tag):
        think_support = True
        thinking = True
        body_start = False

        chunk = chunk[len(think_start_tag):].strip()
    
    if think_support and thinking and think_end_tag in chunk:
        thinking = False
        body_start = True
        chunk = chunk.split(think_end_tag, 1)[1]
    
    if not think_support or body_start:
        chunks = chunk.strip().split(f"```{format}", 1)
        if len(chunks) == 2:
            chunk = chunks[1].split(f"```{format}", 1)[0]
        chunk = chunk.strip().strip("`")
        
        if format == "yaml":
            # YAMLの処理
            try:
                data = yaml.safe_load(chunk)
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list):
                    return data
                else:
                    return {}
            except yaml.YAMLError as e:
                print(e)
                return {}
        elif format == "json":
            # JSONの処理
            try:
                data = loads(chunk)
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list):
                    return data
                else:
                    return {}
            except json.JSONDecodeError as e:
                print(e)
                return {}

    return {}


async def stream_demo_async():
    """非同期版ストリーミングのデモ（リアルタイムJSON処理付き）"""
    import logging

    # メイン処理
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger("chat_assistant")
    logger.setLevel(logging.INFO)

    models = ["lambda/deepseek-v3-0324", "lambda/deepseek-llama3.3-70b", "lambda/deepseek-r1-0528", "gemini/gemini-2.5-pro", ]
    model_manager = ModelManager(models=models)

    # 画面に何秒ごとに表示するか
    timer_sec = 0.2

    import time
    timer_start = time.time()

    async def print_json(responce_text: str):
        result = await chat_stream_data_process(responce_text, format="json")
        print(result)
        print("emotion", result.get("emotion"))
        print("text", result.get("text"))

    responce_text = ""
    async with ChatAssistant(temperature=0.0, model_manager=model_manager) as chat_assistant:
        print("非同期ストリーミング開始:")
        
        async for chunk in chat_assistant.chat_stream(
            message="""
# 命令
次のテキストをJSONに変換して、```json で囲って返してください。
コメントや補足は不要です。

# テキスト
```text
表情は幸せ。
セリフは「ご主人様、お帰りなさいませ。リリエル・アズライト、通称リリと申します。本日もお世話させていただきますね。ご主人様は入室なさいますか、それとも新たに館の主として登録なさいますか？」
```

# 出力
```json
{
    "emotion": "",
    "text": ""
}
```
""".strip()):
            responce_text += chunk

            if timer_start + timer_sec < time.time():
                timer_start = time.time()
                await print_json(responce_text)

        await print_json(responce_text)

if __name__ == "__main__":
    # 同期版のテスト
    # main_sync()

    # 非同期版のテスト
    # asyncio.run(main())
    
    # ストリーミングデモ（同期版）
    #stream_demo_sync()
    
    # ストリーミングデモ（非同期版）
    asyncio.run(stream_demo_async())
    
