# main.py
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

# =========================
# ① モデル設定
# =========================
model = ChatOpenAI(model="gpt-4o-mini")

# =========================
# ② Retrieverの準備
# =========================
langchain_document_retriever = TavilySearchAPIRetriever(k=3).with_config({
    "run_name": "langchain_document_retriever"
})
web_retriever = TavilySearchAPIRetriever(
    k=3,
    search_depth="advanced"
    ).with_config({
    "run_name": "web_retriever"
})

# =========================
# ③ Route定義
# =========================
class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"

class RouteOutput(BaseModel):
    route: Route

# =========================
# ④ Retriever選択用プロンプト
# =========================
route_prompt = ChatPromptTemplate.from_template("""
次の質問に対して、どのRetrieverを使うべきかを選んでください。
選択肢は "langchain_document" または "web" のいずれかです。
出力は JSON形式にしてください。

質問: {question}
""")

# =========================
# ⑤ route_chain（ルーティング）
# =========================
route_chain = (
    route_prompt
    | model.with_structured_output(RouteOutput)
    | (lambda x: x.route)
)

# =========================
# ⑥ Retriever切替関数
# =========================
def routed_retriever(route, question):
    """ルートに応じて適切なRetrieverを実行"""
    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)
    else:
        raise ValueError(f"Unknown route: {route}")

# =========================
# ⑦ 回答生成用プロンプト
# =========================
prompt = ChatPromptTemplate.from_template("""
あなたは優秀なアシスタントです。
次のコンテキストをもとに、質問に対して簡潔でわかりやすく回答してください。

コンテキスト:
{context}

質問:
{question}
""")

# =========================
# ⑧ 全体のChain
# =========================
route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=lambda x: routed_retriever(x["route"], x["question"]))
    | prompt
    | model
    | StrOutputParser()
)

# =========================
# ⑨ メインループ
# =========================
if __name__ == "__main__":
    print("=== RAG Chatbot (qで終了) ===\n")
    while True:
        question = input("質問を入力してください（qで終了）: ").strip()
        if question.lower() == "q":
            print("終了します。")
            break
        try:
            result = route_rag_chain.invoke({"question": question})
            print("\n🧠 回答:")
            print(result)
            print("-" * 50)
        except Exception as e:
            print(f"⚠️ エラーが発生しました: {e}")
            print("-" * 50)
