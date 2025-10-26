import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================================
# 🔹 QueryGenerationOutputモデル定義
# ================================
class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")

# ================================
# 🔹 ベクトルDBをロード
# ================================
def load_vectorstore():
    print("📂 既存の Chroma ベクトルストアを読み込み中...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory="./langchain",  # DBディレクトリ名
        embedding_function=embeddings
    )
    print("✅ ベクトルストアをロードしました。")
    return db

# ================================
# 🔹 複数クエリ生成 + RAGチェーン構築
# ================================
def build_multi_query_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- 複数クエリ生成プロンプト ---
    query_generation_prompt = ChatPromptTemplate.from_template("""
あなたは検索アシスタントです。
以下の質問に対して、ベクターデータベースで関連情報を探すための
3つの異なる検索クエリを生成してください。

- 意味が近いが異なる言い回しを意識する
- 具体的なキーワードや関連語を含める
- 各クエリは20文字以内で簡潔に書く

質問: {question}
""")

    # --- 複数検索クエリ生成チェーン ---
    query_generation_chain = (
        query_generation_prompt
        | model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )

    # --- 検索とRAGプロンプト結合 ---
    rag_prompt = ChatPromptTemplate.from_template("""
次の複数の検索結果をもとに、質問に答えてください。
冗長にならないよう、簡潔でわかりやすく説明してください。

# 検索結果
{context}

# 質問
{question}
""")

    # --- Multi Query RAGチェーン ---
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": query_generation_chain | retriever.map(),  # 各クエリを検索してリストで返す
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )

    return chain

# ================================
# 🔹 実行部分
# ================================
def main():
    db = load_vectorstore()
    rag_chain = build_multi_query_rag_chain(db)

    while True:
        question = input("\n🧠 質問を入力してください（終了: q）> ")
        if question.lower() in ["q", "quit", "exit"]:
            print("👋 終了します。")
            break

        print("💬 複数クエリで検索中...\n")
        answer = rag_chain.invoke(question)
        print("💡 回答:\n", answer)


if __name__ == "__main__":
    main()

