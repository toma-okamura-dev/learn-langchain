import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ================================
# 🔹 複数検索クエリ出力用モデル
# ================================
class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


# ================================
# 🔹 ベクトルDBのロード
# ================================
def load_vectorstore():
    print("📂 既存の Chroma ベクトルストアを読み込み中...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory="./langchain",  # DB保存先
        embedding_function=embeddings
    )
    print("✅ ベクトルストアをロードしました。")
    return db


# ================================
# 🔹 RRF（Reciprocal Rank Fusion）
# ================================
def reciprocal_rank_fusion(
    retriever_outputs: List[List[Document]],
    k: int = 60,
    per_query_cutoff: int = 5,
    top_n: int = 8,
) -> List[str]:
    """
    retriever_outputs: 各クエリで取得した Document のリストのリスト
    k: 平滑化パラメータ（通常 60）
    per_query_cutoff: 各クエリの上位何件を加算対象にするか
    top_n: RRFでスコア付け後に返す上位件数
    """
    key2score: Dict[str, float] = {}
    key2content: Dict[str, str] = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs[:per_query_cutoff]):
            key = (
                doc.metadata.get("doc_id")
                or f'{doc.metadata.get("source")}#{doc.metadata.get("page")}'
                or doc.page_content
            )
            if key not in key2score:
                key2score[key] = 0.0
                key2content[key] = doc.page_content
            key2score[key] += 1.0 / (rank + 1 + k)

    ranked = sorted(key2score.items(), key=lambda kv: kv[1], reverse=True)
    contents = [key2content[key] for key, _ in ranked[:top_n]]
    return contents


# ================================
# 🔹 Multi Query + RRF の RAG-Fusion
# ================================
def build_rag_fusion_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- クエリ生成プロンプト ---
    query_generation_prompt = ChatPromptTemplate.from_template("""
以下の質問に対して、ベクターデータベースから関連情報を検索するための
3つの異なる検索クエリを生成してください。

- 意味が近いが異なる言い回しを意識する
- 具体的な関連語を入れる
- 各クエリは20文字以内で簡潔にする

質問: {question}
""")

    query_generation_chain = (
        query_generation_prompt
        | model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )

    # --- 最終RAGプロンプト ---
    rag_prompt = ChatPromptTemplate.from_template("""
以下の検索結果をもとに質問に答えてください。
冗長にならず、簡潔で正確にまとめてください。

# 検索結果
{context}

# 質問
{question}
""")

    # --- RAG-Fusion チェーン ---
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
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
    rag_chain = build_rag_fusion_chain(db)

    while True:
        question = input("\n🧠 質問を入力してください（終了: q）> ")
        if question.lower() in ["q", "quit", "exit"]:
            print("👋 終了します。")
            break

        print("💬 複数クエリ検索＋RRF融合中...\n")
        answer = rag_chain.invoke(question)
        print("💡 回答:\n", answer)


if __name__ == "__main__":
    main()
