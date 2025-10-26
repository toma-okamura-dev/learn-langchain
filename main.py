import os
from dotenv import load_dotenv
load_dotenv()
import pprint

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# PromptTemplateの作成
prompt = ChatPromptTemplate.from_template(
    """
以下の文脈だけを踏まえて質問に回答してください。
文脈:
{context}

質問: {question}
"""
)

# モデルを準備
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# TavilyをRetrieverとして設定
retriever = TavilySearchAPIRetriever(k=3)

# RAGチェーンを構築
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

output = chain.invoke("今日と明日の五反田の天気について教えてください")
pprint.pprint(output)
