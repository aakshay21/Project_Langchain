from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

print("🔥 LangServe is running...")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Echo & Essay API with LLaMA2"
)

# Route 1: Echo
echo_chain = RunnableLambda(lambda x: {"output": f"You typed: {x['input']}"})
add_routes(app, echo_chain, path="/echo")

# Route 2: Essay (ensure `ollama run llama2` is running in terminal)
llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_template(
    "Write an essay about {topic} in around 100 words."
)

essay_chain = (
    {"topic": lambda x: x["topic"]}
    | prompt
    | llm
    | StrOutputParser()
)

add_routes(app, essay_chain, path="/essay")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
