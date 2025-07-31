import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("🦙 Local LLM Chatbot (LLaMA 2 via Ollama)")

# Input
user_input = st.text_input("Ask me something:")

# Prompt chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}")
])
llm = Ollama(model="llama2")
chain = prompt | llm | StrOutputParser()

# Output
if user_input:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})
        st.markdown(f"**Answer:** {response}")


