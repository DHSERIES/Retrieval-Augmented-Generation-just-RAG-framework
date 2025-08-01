import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI  # You can still use OpenAI LLMs here if desired
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# 1. Load documents
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Use open-source embedding model
embeds = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in Chroma vector store
db = Chroma.from_documents(chunks, embedding=embeds, persist_directory="./chroma_db_open")

# 5. Retrieve relevant chunks
retriever = db.as_retriever(search_kwargs={"k": 3})

# 6. Use OpenAI LLM or any chat model you prefer
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 7. Set up RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Ask a question
query = "What is a large language model?"
result = qa({"query": query})

print("Answer:", result["result"])
print("\nSource documents:")
for doc in result["source_documents"]:
    print("-", doc.page_content[:200], "...")