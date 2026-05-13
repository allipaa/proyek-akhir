import gradio as gr
import requests
import pandas as pd

API_BASE = "http://localhost:8000"

# Sesuai dataset kamu
KATEGORI_LIST = [
    "Troubleshooting",
    "Server & VPS",
    "Billing & Pricing",
    "Network & Security",
    "Performance & Optimization",
    "Layanan Umum"
]

# =========================
# GENERATE KB
# =========================
def generate_draft(kategori, topik):
    # Jika ada topik spesifik, gunakan topik saja agar pencarian lebih akurat.
    # Jika topik kosong, baru gunakan kategori.
    if topik.strip():
        query = topik.strip()
    else:
        query = kategori

    try:
        res = requests.post(
            f"{API_BASE}/generate-kb",
            json={"topic": query},
            timeout=180
        )

        data = res.json()

        if not data.get("success"):
            return "Gagal generate draft", ""

        result = data["data"]

        draft = result.get("draft", "Draft tidak tersedia.")
        referensi = result.get("referensi", [])

        # Format tabel referensi
        if referensi:
            ref_md = f"### 📚 Referensi ({len(referensi)} tiket)\n\n"
            ref_md += "| # | Topik | Kategori | Skor |\n"
            ref_md += "|---|-------|----------|------|\n"

            for i, r in enumerate(referensi, 1):
                ref_md += f"| {i} | {r['subject'][:40]} | {r['kategori']} | {r['score']} |\n"
        else:
            ref_md = "⚠️ Tidak ada tiket relevan ditemukan."

        return draft, ref_md

    except requests.exceptions.ConnectionError:
        return "❌ Backend belum berjalan.\n\nJalankan:\nuvicorn backend.app.main:app --reload --port 8000", ""
    except Exception as e:
        return f"Error: {str(e)}", ""


# =========================
# ANALYZER
# =========================
def load_analysis():
    try:
        res = requests.get(f"{API_BASE}/analyze/top-issues?n=10", timeout=10)
        data = res.json()

        if not data.get("success"):
            return pd.DataFrame({"Error": ["Gagal load data"]})

        df = pd.DataFrame(data["data"])
        df.columns = ["Topik Permasalahan", "Jumlah Tiket"]

        return df

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def load_categories():
    try:
        res = requests.get(f"{API_BASE}/analyze/categories", timeout=10)
        data = res.json()

        if not data.get("success"):
            return pd.DataFrame({"Error": ["Gagal load data"]})

        df = pd.DataFrame(data["data"])
        df.columns = ["Kategori", "Jumlah Tiket"]

        return df

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def load_summary():
    try:
        res = requests.get(f"{API_BASE}/analyze/summary", timeout=30)
        data = res.json()

        if not data.get("success"):
            return "Gagal load summary"

        s = data["data"]

        return f"""
### 📊 Ringkasan Dataset

- Total Tiket       : **{s['total_tiket']}**
- Total Kategori    : **{s['total_kategori']}**
- Kategori Terbanyak: **{s['kategori_terbanyak']}**
- Jumlah            : **{s['jumlah_terbanyak']} tiket**
"""

    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# UI
# =========================
with gr.Blocks(
    title="Knowledge Base Otomatis CloudKilat",
    theme=gr.themes.Soft(primary_hue="blue")
) as app:

    gr.Markdown("""
# 🤖 Kilat-AI: Knowledge Base Generator
## Sistem Generasi Knowledge Base Berbasis Hybrid Retrieval-Augmented Generation (RAG)

Topik apa yang ingin kamu buat menjadi draf artikel hari ini?
""")

    # =========================
    # TAB 1 - GENERATE KB
    # =========================
    with gr.Tab("📝 Generate Knowledge Base"):

        gr.Markdown("### Buat Draft Artikel Knowledge Base")

        with gr.Row():
            with gr.Column(scale=1):
                inp_kat = gr.Dropdown(
                    choices=KATEGORI_LIST,
                    label="Kategori",
                    value="Server & VPS"
                )

                inp_top = gr.Textbox(
                    label="Topik Spesifik (opsional)",
                    lines=4,
                    placeholder="Contoh: server tidak bisa booting"
                )

                btn_gen = gr.Button(" Generate Draft", variant="primary")

                out_ref = gr.Markdown(label="Referensi Tiket")

            with gr.Column(scale=2):
                out_draft = gr.Markdown(
                    label="Draft Artikel",
                    value="Draft akan muncul di sini..."
                )

        btn_gen.click(
            fn=generate_draft,
            inputs=[inp_kat, inp_top],
            outputs=[out_draft, out_ref]
        )

    # =========================
    # TAB 2 - ANALYZER
    # =========================
    with gr.Tab("📊 Analisis Permasalahan"):

        gr.Markdown("### Insight Permasalahan Pelanggan")

        btn_summary = gr.Button("📌 Tampilkan Ringkasan")
        out_summary = gr.Markdown()

        with gr.Row():
            btn_iss = gr.Button("🔍 Top Permasalahan")
            btn_cat = gr.Button("📂 Distribusi Kategori")

        with gr.Row():
            tbl_iss = gr.Dataframe(
                headers=["Topik Permasalahan", "Jumlah Tiket"],
                wrap=True
            )

            tbl_cat = gr.Dataframe(
                headers=["Kategori", "Jumlah Tiket"],
                wrap=True
            )

        btn_summary.click(fn=load_summary, outputs=out_summary)
        btn_iss.click(fn=load_analysis, outputs=tbl_iss)
        btn_cat.click(fn=load_categories, outputs=tbl_cat)


if __name__ == "__main__":
    app.launch(server_name="localhost", server_port=7860, inbrowser=True)