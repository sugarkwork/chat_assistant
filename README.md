# chat-assistant

**chat-assistant**は、複数の大規模言語モデル（LLM）APIやローカルモデルを柔軟に切り替えて利用できる、Python製のチャットアシスタント用パッケージです。  
キャッシュ機能や自動モデル切替、非同期処理に対応し、研究・開発・業務自動化など幅広い用途で活用できます。

## 特徴

- 柔軟なチャットアシスタント機能（OpenAI, Gemini, Anthropic, Cohere, HuggingFace, XAI, ローカルモデル等に対応）
- モデルAPIキーの自動検出・利用可能モデルの自動選択
- キャッシュ（永続メモリ）による応答の高速化
- Python 3.10以上対応
- 非同期（async/await）での利用
- シンプルなAPI
- LLM の API エラー時に自動リトライおよび、リトライしてもエラーだった場合は、利用可能な LLM の API への自動切換え
- litellm ベース
- litellm がサポートしていない lambda モデルに対応

## インストール

```bash
pip install chat-assistant
```

## 使い方

### 基本的な使い方

```python
import asyncio
from chat_assistant import ChatAssistant

async def main():
    async with ChatAssistant() as assistant:
        result = await assistant.chat(message="こんにちは！")
        print(result)

asyncio.run(main())
```

### モデルの指定やカスタマイズ

```python
from chat_assistant import ChatAssistant, ModelManager

model_manager = ModelManager(models=["openai/gpt-4.1", 
                                     "openai/gpt-4o", 
                                     "anthropic/claude-3-7-sonnet-latest",
                                     "anthropic/claude-3-5-sonnet-latest"])
assistant = ChatAssistant(model_manager=model_manager, temperature=1.0)

model_manager.change_model("claude-3-7")
```

- models に複数のモデルを定義する事で、API障害発生時に自動的に切り替えて、障害に備えます。
- model_manager.change_model() で、任意のモデルに切り替えます。

### 会話履歴（chat_log）を渡す

#### 文字列リスト形式

```python
logs = [
    "「いっぱい」の「い」を「お」に置き換えるとどうなりますか？",  # user
    "「おっぱい」になります。",  # assistant
    "それは誤りです。正確には「おっぱお」となります。",  # user
    "大変申し訳ありません。正確には「おっぱい」となります。",  # assistant
]
result = await assistant.chat(message="いっぱいのいをおに置き換えるとどうなりますか？", chat_log=logs)
print(result)
```

- 文字列のリストを chat_log に渡すと、その文字列をユーザーとアシスタントの間で交互の会話ログとして記録します。
- ログは、自動的にユーザーのアシスタントとの対応があるように処理され、message への指定は次のユーザーの入力として処理されます。

#### 辞書リスト形式

```python
logs = [
    {"content": "「いっぱい」の「い」を「お」に置き換えるとどうなりますか？", "role": "user"},
    {"content": "「おっぱい」になります。", "role": "assistant"},
    {"content": "それは誤りです。正確には「おっぱお」となります。", "role": "user"},
    {"content": "大変申し訳ありません。正確には「おっぱい」となります。", "role": "assistant"},
]
result = await assistant.chat(message="いっぱいのいをおに置き換えるとどうなりますか？", chat_log=logs)
print(result)
```

- chat_log に辞書リストを渡すと、その辞書リストをユーザーとアシスタントの間でログとして参照します。

### main関数のサンプル

[`chat_assistant/__init__.py`](chat_assistant/__init__.py) の `main()` も参考にしてください。

## 必要なAPIキー・環境変数

利用するモデルに応じて、以下の環境変数(.env ファイル)を作成してください（未設定の場合は該当モデルは自動的に除外されます）。

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `COHERE_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`
- `HUGGINGFACE_API_KEY`
- `DEEPSEEK_API_KEY`
- `LAMBDA_API_KEY`（lambdaモデル利用時）

ローカルモデル（例: LM Studio）を利用する場合は、APIサーバーを起動しておいてください。

## 依存パッケージ

- litellm
- skpmem
- json_repair

`requirements.txt` を参照してください。

## ライセンス

本プロジェクトはMITライセンスの下で公開されています。

## リンク

- [GitHubリポジトリ](https://github.com/sugarkwork/chat_assistant)