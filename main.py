from chunking import chunk_documents
from dataset import load_raw_dataset
from ingest import load_documents


def main():
    # RAG pipeline:
    # 1 Load raw dataset
    rows = load_raw_dataset()
    # 2. Prepare for ingestion converting to LangChain `Document`s
    docs = load_documents(rows, scope="subset")
    print(f"Loaded {len(docs)} documents")
    # print the first document to verify
    print(docs[0])
    # 3. Split into retrievable chunks
    chunks = chunk_documents(docs, chunk_strategy="recursive", scope="subset")
    print(f"Chunked into {len(chunks)} chunks")



if __name__ == "__main__":
    main()
