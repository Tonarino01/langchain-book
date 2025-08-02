import os
import time
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# LangChainの各種コンポーネントをインポート
from langchain.chat_models import ChatOpenAI                     # OpenAIのChatモデルを使う
from langchain.schema import HumanMessage                       # ユーザー入力をメッセージ形式で扱う
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler        # LangChainの処理ログをStreamlitに出力するためのコールバック
from langchain.memory import ConversationBufferMemory           # 会話履歴を保持するためのMemory（リスト型）
from langchain.prompts import MessagesPlaceholder               # プロンプトテンプレートに履歴挿入位置をマークするための構成要素

# .envファイルを自動検出・読み込みしてAPIキーや設定を環境変数として登録する
load_dotenv(find_dotenv())

# Streamlitのタイトル設定（画面上部に表示）
st.title("langchain-streamlit-app")

# 会話エージェントの構築を行う関数
def create_agent_chain():
    # ChatOpenAIインスタンスの生成：LLM（大規模言語モデル）
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],         # 例: "gpt-3.5-turbo" or "gpt-4"
        temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),  # 出力のランダム性（floatに変換が必要）
        streaming=True                                     # 応答をトークン単位でストリーミング出力（UIではCallbackHandlerで処理）
    )

    # 使用可能な外部ツール（LangChain標準）をロード
    tools = load_tools(["ddg-search", "wikipedia"])  # DuckDuckGoとWikipediaを検索ツールとして使用

    # 会話履歴の保存を担当するMemoryオブジェクト
    memory = ConversationBufferMemory(
        memory_key="memory",          # 会話履歴のキー名 → プレースホルダーと一致させる必要あり
        return_messages=True          # Chat形式での履歴（HumanMessage, AIMessage）として返す（必須）
    )

    # プロンプトテンプレートに「履歴を差し込む場所」を指示するマーカー
    agent_kwargs = {
        "extra_prompt_messages": [    # ここでテンプレートに履歴挿入位置を明示する・・・とは言え AgentType を指定するのは初心者向けの構成であり、裏で定められた箇所に挿入されることとなる。
            MessagesPlaceholder(variable_name="memory")  # "memory" は memory_key と一致する必要あり
        ],
    }

    # LangChainのエージェント初期化（LLM＋Tool＋Memory＋PromptTemplateの統合体を生成）
    return initialize_agent(
        tools=tools,                             # 使用するツールリスト
        llm=chat,                                # OpenAIのチャットモデル（ChatOpenAI）
        memory=memory,                           # 会話履歴オブジェクト
        agent_kwargs=agent_kwargs,               # 履歴の差し込み位置のテンプレート指定
        agent=AgentType.OPENAI_FUNCTIONS         # Function CallingベースのAgent（Tool自動選択可）
    )

# セッションステートにエージェントが存在しない場合、初期化して保存する
# Streamlitは毎回コードを再実行するため、状態の永続化には session_state を使う
if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()  # 一度だけエージェントを構築

# チャット履歴（UI表示用）を初期化：今回のこのコードにおいてはLangChainのMemoryとは別管理にしている。（本当は一括にしてズレが生じないように調整すべき。）
if "messages" not in st.session_state:
    st.session_state.messages = []  # Streamlit上のチャット表示に使うメッセージ履歴（辞書形式）

# すでに保存されているメッセージ履歴を表示
# LangChainの履歴（memory）とは別に、画面表示用の履歴を独立して持っている点に注意
for message in st.session_state.messages:
    with st.chat_message(message["role"]):       # "user" または "assistant" が入りうる。
        st.markdown(message["content"])          # Markdown形式で表示（改行・強調なども使える）

# ユーザーの入力を受付（画面下部のチャット入力欄）
prompt = st.chat_input("What is up?")

if prompt:
    # ユーザーの発言を表示履歴に追加（UI用）
    st.session_state.messages.append({"role": "user", "content": prompt})

    # チャット画面にユーザーの吹き出しを表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # DuckDuckGoなどのツールがレート制限されるのを防ぐために少し待機（簡易的な対策）
    time.sleep(5)

    with st.chat_message("assistant"):
        # LangChainの実行ログ（ツール選択や思考過程）をStreamlitのUI上に表示するためのコールバックハンドラ
        callback = StreamlitCallbackHandler(st.container())  # st.container() にログが出る
        # エージェントを実行（ユーザー入力を渡す）
        response = st.session_state.agent_chain.run(
            prompt,                    # ユーザーの入力内容
            callbacks=[callback]      # 実行過程をUIにストリーム表示するためのハンドラ
        )
        # LLMによる最終的な応答をMarkdown形式で表示
        st.markdown(response)

    # 応答もUI履歴に追加（会話の流れを維持するため）
    st.session_state.messages.append({"role": "assistant", "content": response})
