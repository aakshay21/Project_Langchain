from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
from langchain_community.embeddings import HuggingFaceEmbeddings


##Load Embeddings and FAISS Index
embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("internal_document_chatbot/faiss_index_recursive", embeddings=embedding, allow_dangerous_deserialization=True)


# Dynamic filter function generator based on the query content
def boosted_filter(query: str):
    keywords = []
    query = query.lower()

    if "maternity" in query:
        keywords.append("maternity")
    if any(k in query for k in ["vpn", "security", "es plus"]):
        keywords.append("wfh_security")
    if any(k in query for k in ["camera", "video"]):
        keywords.append("wfh_camera")
    if any(k in query for k in ["apply", "short", "request", "hrms"]):
        keywords.append("wfh_short_term")

    def filter_fn(metadata):
        tag = metadata.get("tag", "")
        return tag in keywords if keywords else True  # fallback allow all

    return filter_fn




##query
query = "I want to apply for wfh?"

##Dynamically fenerate the metadata filter based on the query
filter_fn = boosted_filter(query)

##Convert to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20, 'filter':boosted_filter(query)}
    )


#Define both prompts
## Map Prompt: run on individual chunks
map_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an internal HR assistant.

Use the following context to answer the question strictly **based on the information provided**.
Include all **important factual details** from the context that are relevant to the question.
If multiple policies apply (e.g., dual employment, reapplication, exceptions), include all relevant ones. Be accurate and mention specific constraints if present.

Avoid assumptions. Only respond using specific instructions or policy steps. Do not assume or generalize.


<context>
{context}
</context>

Question:
{question}

Answer:
"""
)



## Reduce Prompt: combines the mapped answers
reduce_prompt = PromptTemplate.from_template(
   """
You are a strict HR policy assistant. Use only the provided context to answer the user's question. Do NOT rely on outside knowledge.

Question: {question}

Context:
{context}

Guidelines:
- If the answer is directly in the context, answer concisely.
- If the answer is implied or scattered across context, summarize it faithfully.
- If the answer is NOT present, say: "The policy document does not mention this explicitly."

Return only the answer.
"""

)







##Load model
llm = Ollama(model="llama2", temperature=0)


##Set up RetrievalQA with map_reduce
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce",
    chain_type_kwargs={
        "question_prompt": map_prompt,
        "combine_prompt": reduce_prompt
    },
    return_source_documents=True
    #verbose = True
)


##Run query
query = "Which documents are needed in a WFH request?"

print("🧠 Running QA for:", query)
response = qa_chain.invoke({"query": query})

print("\n✅ Answer:\n", response["result"] if isinstance(response, dict) else response)

for i, doc in enumerate(response["source_documents"]):
    print(f"\n📄 Chunk {i+1} (Tag: {doc.metadata.get('tag')})")
    print(doc.page_content[:500])

