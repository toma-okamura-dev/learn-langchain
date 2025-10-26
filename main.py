import os
from dotenv import load_dotenv
load_dotenv()
import pprint

from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Step 1: .md ファイルをGitHubから読み込む ---
def file_filter(file_path: str) -> bool:
    # MarkdownとPythonファイルを対象にする
    return file_path.endswith(".py") or file_path.endswith(".md")


def load_documents():
    print("GitHubリポジトリからドキュメントを読み込み中...")
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter,
    )
    documents = loader.load()
    print(f"読み込んだドキュメント数: {len(documents)}")
    return documents


# --- Step 2: OpenAIのEmbeddingモデルでベクトル化 & Chromaへ格納 ---
def build_vectorstore(documents):
    print("OpenAI Embeddings でベクトル化中...")

    # 1) まず文書をチャンク分割（サイズとオーバーラップは用途に応じて調整）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # ≒ 1000 文字ベース（トークンで厳密にやるなら TokenTextSplitter を使用）
        chunk_overlap=200,
        length_function=len,  # まずはシンプルに。厳密にトークン管理したければ tiktoken を使う
        add_start_index=True,
    )
    split_docs = splitter.split_documents(documents)
    print(f"分割後のチャンク数: {len(split_docs)}")

    # 2) 埋め込み器を用意
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3) 空の Chroma を作ってから「小分けで」追加
    db = Chroma(embedding_function=embeddings)

    # バッチを小さく保つ（合計トークンが30万を超えない目安として、まずは100〜200件ずつ）
    batch_size = 100
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i:i + batch_size]
        db.add_documents(batch)
        if (i // batch_size) % 10 == 0:
            print(f"  進捗: {i + len(batch)}/{len(split_docs)} チャンクを追加")

    print("Chromaベクトルストアを作成しました。")
    return db


# --- Step 3: RAG チェーンを構築 ---
def build_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # プロンプトテンプレート
    prompt = ChatPromptTemplate.from_template(
        """以下の文脈だけを踏まえて質問に回答してください。
文脈:
{context}

質問: {question}
"""
    )

    # LLM設定
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # チェーン構成
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


# --- Step 4: 実行部分 ---
def main():
    # ドキュメントを読み込み
    documents = load_documents()

    if len(documents) == 0:
        print("⚠️ ドキュメントが見つかりませんでした。ファイルフィルタを確認してください。")
        return

    # ベクトルストア作成
    db = build_vectorstore(documents)

    # RAGチェーン作成
    rag_chain = build_rag_chain(db)

    # 質問してみる
    question = "LangChainの概要を教えて"
    print(f"\n🧠 質問: {question}")
    answer = rag_chain.invoke(question)
    print("\n💬 回答:\n", answer)


if __name__ == "__main__":
    main()
