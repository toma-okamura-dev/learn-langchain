import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# モデルと出力パーサ
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

# 1本目: ステップバイステップで考えさせる
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問にステップバイステップで回答してください。"),
    ("human", "{question}")
])
cot_chain = cot_prompt | model | output_parser

# 2本目: 結論だけ抽出
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "次の回答から結論だけを抽出してください。"),
    ("human", "{text}")
])
summarize_chain = {"text": RunnablePassthrough()} | summarize_prompt | model | output_parser

# 連結
cot_summarize_chain = cot_chain | summarize_chain

# 実行
result = cot_summarize_chain.invoke({"question": "10 + 2 * 3 はいくつ？"})
print(result)
