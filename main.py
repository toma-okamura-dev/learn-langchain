import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- 既存のベクトルDBをロード ---
def load_vectorstore():
    print("📂 既存の Chroma ベクトルストアを読み込み中...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory="./langchain",  # 保存したDBのディレクトリ名に合わせて変更
        embedding_function=embeddings
    )
    print("✅ ベクトルストアをロードしました。")
    return db


# --- 質問用チェーンを構築 ---
def build_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """以下の文脈に基づいて質問に答えてください。

文脈:
{context}

質問: {question}
"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain


# --- 実行部分 ---
def main():
    db = load_vectorstore()
    rag_chain = build_rag_chain(db)

    while True:
        question = input("\n🧠 質問を入力してください（終了: q）> ")
        if question.lower() in ["q", "quit", "exit"]:
            print("👋 終了します。")
            break

        print("💬 回答を生成中...\n")
        answer = rag_chain.invoke(question)
        print("💡 回答:\n", answer)


if __name__ == "__main__":
    main()
