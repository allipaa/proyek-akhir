import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from .config import (
    CHROMA_DIR, TICKET_FILE, EMBEDDING_MODEL, COLLECTION_NAME,
    RETRIEVER_K, BM25_K, WEIGHT_SEMANTIC, WEIGHT_BM25, SIMILARITY_THRESHOLD
)

# Variabel Global agar tidak load ulang terus (Singleton)
_ensemble_retriever = None
_cached_docs = None 

def _build_docs_from_csv() -> list[Document]:
    global _cached_docs
    if _cached_docs is not None:
        return _cached_docs
        
    # Menggunakan sep='\t' sesuai dataset kamu
    df = pd.read_csv(TICKET_FILE, sep='\t')
 
    docs = []
    for _, row in df.iterrows():
        subject = str(row.get("subject", ""))
        content = str(row.get("content", ""))
        text = f"{subject}\n{content}".strip()
        
        if text:
            docs.append(Document(
                page_content=text,
                metadata={
                    "subject"   : subject,
                    "ticket_id" : str(row.get("ticket_id", "-")),
                    "kategori"  : str(row.get("kategori", "-")),
                    "score"     : str(row.get("quality_score", "0")), 
                }
            ))
    _cached_docs = docs
    return docs

def _get_ensemble_retriever() -> EnsembleRetriever:
    global _ensemble_retriever
    if _ensemble_retriever is not None:
        return _ensemble_retriever

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding,
        collection_name=COLLECTION_NAME
    )
    
    # Gunakan similarity_search agar bisa memproses query
    semantic_retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVER_K,
            "score_threshold": SIMILARITY_THRESHOLD 
        }
    )

    docs = _build_docs_from_csv()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = BM25_K

    _ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[WEIGHT_SEMANTIC, WEIGHT_BM25]
    )
    return _ensemble_retriever

def retrieve_tickets(query: str) -> list[dict]:
    # 1. Panggil global ensemble retriever (Hybrid: BM25 + Semantic)
    ensemble_retriever = _get_ensemble_retriever()
    
    # 2. Ambil dokumen hasil hybrid search
    docs = ensemble_retriever.invoke(query)
 
    tickets = []
    seen_ids = set()
    

    for doc in docs:
        meta = doc.metadata
        ticket_id = meta.get("ticket_id", "-")
        
        # Ambil quality_score dari metadata (berasal dari kolom quality_score di CSV)
        try:
            q_score = float(meta.get("score", "0"))
        except:
            q_score = 0.0

        # --- LOGIKA FILTER ---
        
        if q_score < 0.1: # Filter (skor 0)
            continue

        if ticket_id in seen_ids:
            continue
        seen_ids.add(ticket_id)

        tickets.append({
            "content"   : doc.page_content,
            "subject"   : meta.get("subject", "-"),
            "ticket_id" : ticket_id,
            "kategori"  : meta.get("kategori", "-"),
            "score"     : str(q_score), 
        })
    
    # 3. Jika setelah difilter hasilnya kosong, 
    # maka list tickets akan kosong [] dan memicu penolakan di kb_generator.
    return sorted(tickets, key=lambda x: float(x['score']), reverse=True)

def retrieve_as_context(query: str) -> tuple[str, list]:
    # Jika query tidak nyambung, retrieve_tickets akan mengembalikan list kosong []
    tickets = retrieve_tickets(query)
    
    if not tickets:
        # Mengembalikan string kosong agar Gemini tahu tidak ada referensi (Fallback)
        return "", []

    parts = []
    referensi = []
    for i, t in enumerate(tickets, 1):
        parts.append(f"[Referensi {i}]\nTopik: {t['subject']}\n{t['content'][:1500]}")
        referensi.append(t) 

    context = "\n\n---\n\n".join(parts)
    return context, referensi