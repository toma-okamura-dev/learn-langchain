import os
from dotenv import load_dotenv
load_dotenv()
import pprint
# main.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
import pprint

# ============================================
# モデルと出力パーサーの準備
# ============================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

# ============================================
# 肯定的な意見を生成するChain
# ============================================
pro_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは政策やテクノロジーに肯定的な立場の評論家です。"),
    ("human", "{topic} に対して、肯定的な意見を述べてください。")
])
pro_chain = pro_prompt | model | output_parser

# ============================================
# 否定的な意見を生成するChain
# ============================================
con_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは批判的な立場の評論家です。"),
    ("human", "{topic} に対して、否定的な意見を述べてください。")
])
con_chain = con_prompt | model | output_parser

# ============================================
# RunnableParallel で並列実行するChain
# ============================================
debate_parallel = RunnableParallel({
    "pro_opinion": pro_chain,
    "con_opinion": con_chain
})

# ============================================
# 並列Chainの実行例
# ============================================
topic_input = {"topic": "AIによる自動運転の普及"}
parallel_output = debate_parallel.invoke(topic_input)

print("=== 肯定派と否定派の意見 ===")
pprint.pprint(parallel_output)

# ============================================
# 意見をまとめるChain（中立的視点）
# ============================================
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは中立的なAIアナリストです。以下の2つの立場をバランスよく要約してください。"),
    ("human", "肯定派の意見: {pro_opinion}\n否定派の意見: {con_opinion}")
])

# ============================================
# RunnableParallel の出力をまとめChainに接続
# ============================================
summary_chain = debate_parallel | summary_prompt | model | output_parser

# ============================================
# 中立的なまとめを出力
# ============================================
final_summary = summary_chain.invoke(topic_input)

print("\n=== 中立的なまとめ ===")
print(final_summary)
