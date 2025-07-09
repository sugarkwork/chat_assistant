import unittest
import os
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
sys.path.insert(0, '/mnt/f/ai/chat_assistant')

# モックモジュールを定義
import json

def loads(text, *args, **kwargs):
    """json_repair.loads関数のモック"""
    return json.loads(text)

class MockPersistentMemory:
    def __init__(self, db_path):
        self.db_path = db_path
        self.data = {}
        
    async def load(self, key):
        return self.data.get(key)
        
    async def save(self, key, value):
        self.data[key] = value
        
    def load_sync(self, key):
        return self.data.get(key)
        
    def save_sync(self, key, value):
        self.data[key] = value
        
    async def close(self):
        pass
        
    def close_sync(self):
        pass

import types
json_repair_module = types.ModuleType('json_repair')
json_repair_module.loads = loads
sys.modules['json_repair'] = json_repair_module

skpmem_module = types.ModuleType('skpmem')
skpmem_module.PersistentMemory = MockPersistentMemory
sys.modules['skpmem'] = skpmem_module

from chat_assistant import ModelManager, ChatAssistant


class TestModelManager(unittest.TestCase):
    """ModelManagerクラスのテストケース"""
    
    def test_init_default_models(self):
        """デフォルトモデルの初期化テスト"""
        manager = ModelManager(auto_remove_models=False)
        self.assertGreater(len(manager.models), 0)
        self.assertEqual(manager.current_model_index, 0)
        
    def test_init_custom_models(self):
        """カスタムモデルでの初期化テスト"""
        custom_models = ["test/model1", "test/model2"]
        manager = ModelManager(models=custom_models, auto_remove_models=False)
        self.assertEqual(manager.models, custom_models)
        
    def test_get_current_model(self):
        """現在のモデル取得テスト"""
        models = ["test/model1", "test/model2"]
        manager = ModelManager(models=models, auto_remove_models=False)
        self.assertEqual(manager.get_current_model(), "test/model1")
        
    def test_change_model_exact_match(self):
        """モデル変更テスト（完全一致）"""
        models = ["test/model1", "test/model2"]
        manager = ModelManager(models=models, auto_remove_models=False)
        
        result = manager.change_model("test/model2")
        self.assertEqual(result, "test/model2")
        self.assertEqual(manager.current_model_index, 1)
        
    def test_change_model_partial_match(self):
        """モデル変更テスト（部分一致）"""
        models = ["openai/gpt-4", "anthropic/claude-3"]
        manager = ModelManager(models=models, auto_remove_models=False)
        
        result = manager.change_model("gpt")
        self.assertEqual(result, "openai/gpt-4")
        self.assertEqual(manager.current_model_index, 0)
        
    def test_change_model_no_match(self):
        """モデル変更テスト（一致なし）- 例外が発生することを確認"""
        models = ["test/model1", "test/model2"]
        manager = ModelManager(models=models, auto_remove_models=False)
        
        # 見つからない場合は例外が発生
        with self.assertRaises(ValueError):
            manager.change_model("nonexistent")
        
    def test_next_model(self):
        """次のモデルへの切り替えテスト"""
        models = ["test/model1", "test/model2", "test/model3"]
        manager = ModelManager(models=models, auto_remove_models=False)
        
        manager.next_model()
        self.assertEqual(manager.current_model_index, 1)
        
        manager.next_model()
        self.assertEqual(manager.current_model_index, 2)
        
        # 最後のモデルから最初に戻る
        manager.next_model()
        self.assertEqual(manager.current_model_index, 0)
        
    def test_next_model_empty_list(self):
        """空のモデルリストでの次のモデル切り替えテスト"""
        # 強制的に空リストにする
        manager = ModelManager(models=["dummy"], auto_remove_models=False)
        manager.models = []  # 直接空にする
        
        # 空のリストでValueErrorが発生することを確認
        with self.assertRaises(ValueError):
            manager.next_model()
            
    def test_get_current_model_empty_list(self):
        """空のモデルリストでの現在モデル取得テスト"""
        # 強制的に空リストにする
        manager = ModelManager(models=["dummy"], auto_remove_models=False)
        manager.models = []  # 直接空にする
        
        # 空のリストでValueErrorが発生することを確認
        with self.assertRaises(ValueError):
            manager.get_current_model()


class TestChatAssistant(unittest.TestCase):
    """ChatAssistantクラスのテストケース"""
    
    def test_init_default(self):
        """デフォルト初期化テスト"""
        assistant = ChatAssistant()
        self.assertIsNotNone(assistant.model_manager)
        self.assertIsNotNone(assistant.memory)
        self.assertTrue(assistant._local_memory)
        
    def test_init_with_model_manager(self):
        """ModelManagerを指定した初期化テスト"""
        model_manager = ModelManager(models=["test/model"], auto_remove_models=False)
        assistant = ChatAssistant(model_manager=model_manager)
        self.assertEqual(assistant.model_manager, model_manager)
        
    def test_build_messages_system_only(self):
        """システムメッセージのみのメッセージ構築テスト"""
        assistant = ChatAssistant()
        messages = assistant._build_messages("system prompt", "user message")
        
        expected = [
            {"content": "system prompt", "role": "system"},
            {"content": "user message", "role": "user"}
        ]
        self.assertEqual(messages, expected)
        
    def test_build_messages_no_system(self):
        """システムメッセージなしのメッセージ構築テスト"""
        assistant = ChatAssistant()
        messages = assistant._build_messages("", "user message")
        
        expected = [
            {"content": "user message", "role": "user"}
        ]
        self.assertEqual(messages, expected)
        
    def test_build_messages_with_dict_chat_log(self):
        """辞書形式のチャットログを含むメッセージ構築テスト"""
        assistant = ChatAssistant()
        chat_log = [
            {"content": "previous user message", "role": "user"},
            {"content": "previous assistant message", "role": "assistant"}
        ]
        messages = assistant._build_messages("system", "current message", chat_log)
        
        expected = [
            {"content": "system", "role": "system"},
            {"content": "previous user message", "role": "user"},
            {"content": "previous assistant message", "role": "assistant"},
            {"content": "current message", "role": "user"}
        ]
        self.assertEqual(messages, expected)
        
    def test_build_messages_with_string_list_chat_log(self):
        """文字列リスト形式のチャットログを含むメッセージ構築テスト"""
        assistant = ChatAssistant()
        chat_log = ["user message 1", "assistant message 1", "user message 2"]
        messages = assistant._build_messages("", "current message", chat_log)
        
        expected = [
            {"content": "user message 1", "role": "user"},
            {"content": "assistant message 1", "role": "assistant"},
            {"content": "user message 2", "role": "user"},
            {"content": "current message", "role": "user"}
        ]
        self.assertEqual(messages, expected)
        
    def test_build_messages_with_string_chat_log(self):
        """文字列形式のチャットログを含むメッセージ構築テスト"""
        assistant = ChatAssistant()
        chat_log = "previous message"
        messages = assistant._build_messages("", "current message", chat_log)
        
        expected = [
            {"content": "previous message", "role": "user"},
            {"content": "current message", "role": "user"}
        ]
        self.assertEqual(messages, expected)
        
    def test_build_messages_empty_chat_log_list(self):
        """空のチャットログリストでのメッセージ構築テスト"""
        assistant = ChatAssistant()
        messages = assistant._build_messages("", "current message", [])
        
        expected = [
            {"content": "current message", "role": "user"}
        ]
        self.assertEqual(messages, expected)
        
    def test_build_messages_invalid_chat_log_type(self):
        """無効なチャットログタイプでのメッセージ構築テスト"""
        assistant = ChatAssistant()
        
        with self.assertRaises(ValueError):
            assistant._build_messages("", "message", 123)
            
    def test_build_messages_invalid_chat_log_list_type(self):
        """無効なチャットログリストタイプでのメッセージ構築テスト"""
        assistant = ChatAssistant()
        
        with self.assertRaises(ValueError):
            assistant._build_messages("", "message", [123, 456])
            
    def test_parse_result_plain_text(self):
        """プレーンテキストの結果パーステスト"""
        assistant = ChatAssistant()
        result = assistant._parse_result("simple text", False)
        self.assertEqual(result, "simple text")
        
    def test_parse_result_with_think_tags(self):
        """thinkタグを含む結果のパーステスト"""
        assistant = ChatAssistant()
        result = assistant._parse_result("<think>thinking</think>actual response", False)
        self.assertEqual(result, "actual response")
        
    def test_parse_result_json_mode(self):
        """JSONモードでの結果パーステスト"""
        assistant = ChatAssistant()
        result = assistant._parse_result('{"key": "value"}', True)
        self.assertEqual(result, {"key": "value"})
        
    def test_parse_result_json_mode_with_think_tags(self):
        """thinkタグとJSONモードを組み合わせた結果パーステスト"""
        assistant = ChatAssistant()
        result = assistant._parse_result('<think>thinking</think>{"key": "value"}', True)
        self.assertEqual(result, {"key": "value"})
    
    def test_build_messages_empty_chat_log_with_empty_first_element(self):
        """空の最初の要素を持つチャットログでの境界チェックテスト"""
        assistant = ChatAssistant()
        # 空リストの場合、インデックスアクセスでエラーが発生しないことを確認
        chat_log = []
        messages = assistant._build_messages("", "message", chat_log)
        expected = [{"content": "message", "role": "user"}]
        self.assertEqual(messages, expected)
    
    def test_build_messages_single_dict_in_list(self):
        """単一の辞書を含むリストでの境界チェックテスト"""
        assistant = ChatAssistant()
        chat_log = [{"content": "test", "role": "user"}]
        messages = assistant._build_messages("", "message", chat_log)
        expected = [
            {"content": "test", "role": "user"},
            {"content": "message", "role": "user"}
        ]
        self.assertEqual(messages, expected)
    
    def test_build_messages_single_string_in_list(self):
        """単一の文字列を含むリストでの境界チェックテスト"""
        assistant = ChatAssistant()
        chat_log = ["test message"]
        messages = assistant._build_messages("", "message", chat_log)
        expected = [
            {"content": "test message", "role": "user"},
            {"content": "message", "role": "user"}
        ]
        self.assertEqual(messages, expected)


class TestChatAssistantStreaming(unittest.TestCase):
    """ChatAssistantクラスのストリーミング機能テストケース"""
    
    def setUp(self):
        """テスト前の準備"""
        # モックレスポンスチャンクを作成
        self.mock_chunks = [
            self._create_chunk("Hello"),
            self._create_chunk(" "),
            self._create_chunk("world"),
            self._create_chunk("!", finish_reason="stop")
        ]
        
    def _create_chunk(self, content=None, finish_reason=None):
        """テスト用チャンクを作成"""
        chunk = Mock()
        chunk.choices = [Mock()]
        chunk.choices[0].delta = Mock()
        chunk.choices[0].delta.content = content
        chunk.choices[0].finish_reason = finish_reason
        return chunk
    
    def test_chat_stream_sync_basic(self):
        """同期ストリーミングの基本テスト"""
        assistant = ChatAssistant()
        
        # _call_model_stream_syncメソッドをモック
        with patch.object(assistant, '_call_model_stream_sync', return_value=iter(self.mock_chunks)):
            result = []
            for chunk in assistant.chat_stream_sync(message="test"):
                result.append(chunk)
            
            # 各チャンクが正しく返されることを確認
            self.assertEqual(result, ["Hello", " ", "world", "!"])
    
    def test_chat_stream_async_basic(self):
        """非同期ストリーミングの基本テスト"""
        async def run_test():
            assistant = ChatAssistant()
            
            # 非同期イテレータのモック
            async def async_chunks():
                for chunk in self.mock_chunks:
                    yield chunk
            
            # _call_model_streamメソッドをモック
            with patch.object(assistant, '_call_model_stream', return_value=async_chunks()):
                result = []
                async for chunk in assistant.chat_stream(message="test"):
                    result.append(chunk)
                
                # 各チャンクが正しく返されることを確認
                self.assertEqual(result, ["Hello", " ", "world", "!"])
        
        # 非同期テストを実行
        asyncio.run(run_test())
    
    def test_chat_stream_sync_empty_content(self):
        """空コンテンツのチャンク処理テスト（同期）"""
        assistant = ChatAssistant()
        
        # 空コンテンツを含むチャンク
        chunks_with_empty = [
            self._create_chunk("Hello"),
            self._create_chunk(None),  # 空のコンテンツ
            self._create_chunk("world"),
            self._create_chunk("", finish_reason="stop")  # 空文字列
        ]
        
        with patch.object(assistant, '_call_model_stream_sync', return_value=iter(chunks_with_empty)):
            result = []
            for chunk in assistant.chat_stream_sync(message="test"):
                result.append(chunk)
            
            # 空コンテンツがスキップされることを確認
            self.assertEqual(result, ["Hello", "world"])
    
    def test_chat_stream_sync_error_handling(self):
        """ストリーミング中のエラーハンドリングテスト（同期）"""
        manager = ModelManager(models=["test/model1", "test/model2"], auto_remove_models=False)
        assistant = ChatAssistant(model_manager=manager)
        
        # 最初のモデルでエラー、2番目で成功
        def side_effect(*args, **kwargs):
            if assistant.model_manager.current_model_index == 0:
                raise Exception("Model error")
            return iter(self.mock_chunks)
        
        with patch.object(assistant, '_call_model_stream_sync', side_effect=side_effect):
            result = []
            for chunk in assistant.chat_stream_sync(message="test"):
                result.append(chunk)
            
            # エラー後に次のモデルで成功することを確認
            self.assertEqual(result, ["Hello", " ", "world", "!"])
            self.assertEqual(assistant.model_manager.current_model_index, 1)
    
    def test_chat_stream_async_with_chat_log(self):
        """チャットログを含む非同期ストリーミングテスト"""
        async def run_test():
            assistant = ChatAssistant()
            
            chat_log = [
                {"content": "Previous message", "role": "user"},
                {"content": "Previous response", "role": "assistant"}
            ]
            
            async def async_chunks():
                for chunk in self.mock_chunks:
                    yield chunk
            
            with patch.object(assistant, '_call_model_stream', return_value=async_chunks()) as mock_call:
                result = []
                async for chunk in assistant.chat_stream(message="test", chat_log=chat_log):
                    result.append(chunk)
                
                # メッセージが正しく構築されることを確認
                called_messages = mock_call.call_args[0][1]
                self.assertEqual(len(called_messages), 3)
                self.assertEqual(called_messages[-1]["content"], "test")
        
        # 非同期テストを実行
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()