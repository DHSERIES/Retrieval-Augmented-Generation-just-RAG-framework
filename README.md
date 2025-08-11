# Retrieval-Augmented-Generation-framework

This guide outlines the typical steps to build a RAG pipeline using LangChain.

---

##  1.Load Documents

Import your documents into LangChain using a suitable **Document Loader**.
# using loader from https://python.langchain.com/docs/integrations/document_loaders/ for more detail

##  2. Split Document

Using `RecursiveCharacterTextSplitter` in LangChain approach helps preserve context and readability in chunked data.

# 🔧 Best Practices
- **Chunk Size**: 500–2000 characters (or 200–500 tokens)
- **Chunk Overlap**: 10–20% of chunk size
- **Custom Separators**: If your data has a clear structural pattern

# ⚠️ When to Consider Alternatives
| Scenario | Recommended Splitter |
|----------|----------------------|
| Code files | Language-specific splitters (e.g., `PythonCodeSplitter`) |
| Structured data (JSON, CSV, Markdown tables) | Custom or `MarkdownHeaderTextSplitter` |
| Transcripts | Timestamp/speaker-aware splitters |


## 3. Embedding Document

Converts text chunks into **vector representations** (lists of numbers) that capture their meaning and stored in a **Vector Store** for similarity search.

# Ways to Generate Embeddings
- **Using Pre-trained Embedding Models**
- OpenAI: `text-embedding-ada-002` (cheap & effective for most cases)
- ...
- **Using Local / Open-Source Models**
- sentence-transformers, InstructorEmbedding, LlamaIndex

## 4.Choosing a Vector Store

- Local: FAISS, Chroma
- Cloud: Pinecone, Weaviate, Milvus, Qdrant
vector_store = ...

## 5.Retrieve Relevant Chunks from user query

Once your documents are embedded and stored in a vector DB, you can retrieve relevant chunks using different strategies.

# Similarity Search
- **vectorstore.as_retriever**
- for more info ""https://python.langchain.com/docs/how_to/vectorstore_retriever/"" 
# Max Marginal Relevance (MMR)
# Similarity Search with Metadata Filtering**
# Hybrid Search (Vector + Keyword)
# ... still update

## 6. load LLM model

# read LoadLLM.md for more detail

## 7.Set up question-answer chain

# using RetrievalQA.from_chain_type()
# custom build using template
- example
    <!-- template = """Use the following documents to answer the question.If you don't know, say "I don't know" — do not make things up.
.
<!-- Documents:
{context}

Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)--> 

## 8. interface 
<!-- 
query = "What is a large language model?"
result = qa({"query": query}) -->
or
<!-- def ask_question(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.format(context=context, question=question)
    return llm.predict(final_prompt) -->

    
### index
# retriver chema : List[Document]
<!-- from langchain.schema import Document
from langchain.schema import BaseRetriever

class MyCustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        results = my_custom_search_engine(query)
        return [Document(page_content=r["text"], metadata={"id": r["id"]}) for r in results]

    async def _aget_relevant_documents(self, query: str):
        # optional async version
        pass -->