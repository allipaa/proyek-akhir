import pandas as pd
from .config import TICKET_FILE


def _load() -> pd.DataFrame:
    # FIX separator
    df = pd.read_csv(TICKET_FILE, sep='\t')

    # Bersihin data
    df['subject'] = df['subject'].fillna('Tidak diketahui')
    df['kategori'] = df['kategori'].fillna('Lainnya')
    df['quality_score'] = df['quality_score'].fillna(0)
    df['n_emails'] = df['n_emails'].fillna(0)

    return df


# 🔥 1. TOP MASALAH (untuk UI utama)
def get_top_issues(n: int = 10) -> pd.DataFrame:
    """
    Menampilkan top masalah berdasarkan subject
    """
    df = _load()

    top = (
        df.groupby('subject')
        .size()
        .reset_index(name='total_tiket')
        .sort_values('total_tiket', ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    return top


# 🔥 2. DISTRIBUSI KATEGORI (untuk chart)
def get_category_stats() -> pd.DataFrame:
    """
    Distribusi tiket per kategori
    """
    df = _load()

    stats = (
        df.groupby('kategori')
        .size()
        .reset_index(name='jumlah_tiket')
        .sort_values('jumlah_tiket', ascending=False)
        .reset_index(drop=True)
    )

    return stats


# 🔥 3. SUMMARY (untuk dashboard angka)
def get_summary() -> dict:
    df = _load()

    top_kat = df['kategori'].value_counts()

    return {
        "total_tiket": int(len(df)),
        "total_kategori": int(df['kategori'].nunique()),
        "kategori_terbanyak": top_kat.index[0],
        "jumlah_terbanyak": int(top_kat.iloc[0])
    }


def get_top_detailed(n: int = 5) -> pd.DataFrame:
    """
    Top issue + kualitas + intensitas komunikasi
    """
    df = _load()

    result = (
        df.groupby(['subject', 'kategori'])
        .agg(
            total_tiket=('ticket_id', 'count'),
            avg_quality=('quality_score', 'mean'),
            avg_emails=('n_emails', 'mean')
        )
        .reset_index()
        .sort_values('total_tiket', ascending=False)
        .head(n)
    )

    return result