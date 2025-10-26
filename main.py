import os
from dotenv import load_dotenv
load_dotenv()
import pprint

from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
        file_filter_fn=file_filter,
    )
    documents = loader.load()
    print(f"読み込んだドキュメント数: {len(documents)}")
    return documents


# --- Step 2: OpenAIのEmbeddingモデルでベクトル化 & Chromaへ格納 ---
def build_vectorstore(documents):
    print("OpenAI Embeddings でベクトル化中...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(documents, embeddings)
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
