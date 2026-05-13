import pandas as pd
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import TICKET_FILE, CHROMA_DIR, EMBEDDING_MODEL, COLLECTION_NAME

def build_index():
    print("=" * 55)
    print("  MEMBANGUN VECTOR DATABASE (ChromaDB)")
    print("=" * 55)

    if not TICKET_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {TICKET_FILE}")

    # ✅ FIX separator
    df = pd.read_csv(TICKET_FILE, sep='\t')

    print(f"Dataset dimuat: {len(df)} tiket")
    print(f"Kolom         : {df.columns.tolist()}")

    # Validasi kolom
    required_cols = ['content', 'subject', 'ticket_id', 'kategori', 'quality_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset!")

    # 🔥 Gabungkan subject + content
    texts = (df['subject'].fillna('') + " " + df['content'].fillna('')).tolist()

    # Metadata
    metadatas = df[['subject', 'ticket_id', 'kategori', 'quality_score', 'n_emails']] \
        .fillna(0).to_dict(orient='records')

    # Embedding model (LangChain)
    print(f"\nMemuat embedding model: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Reset Chroma
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    print("Menyimpan ke ChromaDB...")

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME
    )

    vectordb.persist()

    print(f"\nChromaDB berhasil dibuat!")
    print(f"  Total dokumen : {len(texts)}")
    print(f"  Lokasi        : {CHROMA_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    build_index()