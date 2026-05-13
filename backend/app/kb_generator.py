from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .retriever import retrieve_as_context
from .config import GEMINI_API_KEY, LLM_MODEL, TEMPERATURE

# agar GEMINI_API_KEY sudah pasti terbaca dari .env sebelum digunakan
def _get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE
    )


# Prompt untuk topik yang RELEVAN dengan dataset
KB_PROMPT = PromptTemplate(
    input_variables=["context", "topic"],
    template="""Anda adalah AI Technical Writer CloudKilat.
TUGAS: Susun artikel Knowledge Base HANYA berdasarkan referensi tiket berikut:
{context}

Topik: {topic}

ATURAN KETAT:
1. Jika referensi di atas tidak menjawab topik secara spesifik, tuliskan: "Maaf, informasi tidak ditemukan dalam dataset."
2. JANGAN gunakan pengetahuan di luar referensi yang diberikan.
3. Langsung mulai ke struktur artikel (Judul, Penyebab, Solusi)."""
)

# Template saat data tidak ditemukan sama sekali (pemicu fallback)
REJECTION_PROMPT = """## Informasi Tidak Ditemukan
Maaf, topik "{topic}" tidak ditemukan dalam histori tiket support CloudKilat.

Sistem tidak dapat menghasilkan draf artikel karena tidak ada referensi data yang valid untuk menjamin akurasi informasi.

---
*Status: Data tidak tersedia dalam dataset internal.*"""

def generate_kb_draft(topic: str) -> dict:
    llm = _get_llm()
    context, referensi = retrieve_as_context(topic)

    # Logika kunci: Jika referensi kosong, langsung lempar prompt penolakan
    if not referensi or len(referensi) == 0:
        final_prompt = REJECTION_PROMPT.format(topic=topic)
        # Langsung return tanpa lewat LLM Gemini untuk hemat kuota & akurasi
        return {
            "topic": topic,
            "draft": final_prompt,
            "referensi": []
        }
    
    # Jika ada data, baru minta Gemini tulis
    final_prompt = KB_PROMPT.format(topic=topic, context=context)
    response = llm.invoke(final_prompt)

    return {
        "topic": topic,
        "draft": response.content,
        "referensi": referensi
    }