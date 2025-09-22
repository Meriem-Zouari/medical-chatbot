from flask import Flask, request, jsonify, render_template
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI  # updated import
from langchain.chains import RetrievalQA  # updated for retrieval
from langchain.prompts import ChatPromptTemplate  # updated import
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings
embedding = download_hugging_face_embeddings()

# Pinecone index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embedding
)

# Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Chat model
chat_model = ChatOpenAI(model="gpt-4o")  # updated variable name to snake_case

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Retrieval chain
# Using RetrievalQA as the current LangChain API
from langchain.chains import RetrievalQA
rag_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")  # safer
    print("User input:", msg)

    response = rag_chain.run(msg)  # corrected invoke/run syntax
    print("Response:", response)
    return str(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
