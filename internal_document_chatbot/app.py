import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from run_qa_chain import qa_chain


##Load vectorstore and embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("internal_document_chatbot/faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":3})


##Custom prompt
prompt_template = PromptTemplate(
    input_variables=["context_str","question"],
    template="""
You are an internal HR and Company policy assistant for an IT based company.
Use the following context to answer the question. Be precise and refer to official policy only.

Context:
{context_str}

Question:
{question}
"""
)

##LLM
llm = Ollama(model="llama2", temperature = 0)


##Chain Setup
##Build Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever,
    chain_type = "refine",
    chain_type_kwargs = {
        "question_prompt":prompt_template,
        "refine_prompt":prompt_template
        }
)


##Streamlit UI
st.set_page_config(page_title = "Internal Policy Chatbot", page_icon="🧠")

st.title("📄 Internal HR Policy Assistant")

query = st.text_input("Ask a question about company policy: ")


if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.invoke({"query":query})
    st.success("Answer: ")
    st.write(answer["result"])