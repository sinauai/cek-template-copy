import streamlit as st
import json
import os
import time
import pathlib
import numpy as np
from openai import OpenAI

# ── Configuration ───────────────────────────────────────────────────────────────
KB_BASE         = "news"                    # will look for news.json or news.txt
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL      = "gpt-4.1"
BATCH_SIZE      = 10                        # embeddings batch size
TRUNCATE_CHARS  = 3000                      # max chars per record

# ── Load OpenAI API key ─────────────────────────────────────────────────────────
api_key = (
    st.secrets.get("openai", {}).get("api_key")
    or os.getenv("OPENAI_API_KEY")
)
if not api_key:
    st.error(
        "🚨 Missing OpenAI API key. Define under [openai] api_key in secrets.toml "
        "or set the OPENAI_API_KEY environment variable."
    )
    st.stop()
client = OpenAI(api_key=api_key)

# ── Embedding helper ────────────────────────────────────────────────────────────
@st.cache_resource
def compute_embeddings(texts: list[str]) -> list[list[float]]:
    all_embeds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        except Exception as e:
            st.error(f"Embedding API error: {e}")
            return []
        all_embeds.extend(d.embedding for d in resp.data)
        time.sleep(0.2)
    return all_embeds

# ── Knowledge loader (JSON or TXT with paragraph splitting) ────────────────────
@st.cache_data
def load_knowledge(base_path: str = KB_BASE) -> list[dict]:
    json_path = f"{base_path}.json"
    txt_path  = f"{base_path}.txt"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict):
            entries = (
                raw.get("Knowledge")
                or raw.get("data")
                or next((v for v in raw.values() if isinstance(v, list)), [])
            )
        else:
            entries = []
    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        # split into paragraphs on blank lines
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        entries = [
            {"title": f"Paragraph {i+1}", "content": para}
            for i, para in enumerate(paras)
        ]
    else:
        st.error(f"No knowledge file found at {json_path} or {txt_path}")
        return []

    records = []
    for idx, rec in enumerate(entries):
        if isinstance(rec, dict):
            title   = rec.get("title") or rec.get("headline") or f"Entry {idx}"
            content = rec.get("content") or rec.get("body")     or json.dumps(rec, ensure_ascii=False)
        else:
            title, content = f"Entry {idx}", str(rec)
        records.append({"id": idx, "title": title, "content": content})
    return records

# ── Build vector store ──────────────────────────────────────────────────────────
@st.cache_resource
def build_vector_store(records: list[dict]):
    texts     = [r["content"][:TRUNCATE_CHARS] for r in records]
    embeddings = compute_embeddings(texts)
    if not embeddings:
        st.stop()
    norms     = [np.linalg.norm(e) for e in embeddings]
    return embeddings, norms

# ── Retriever via cosine similarity ─────────────────────────────────────────────
def retrieve(query: str, records, embeddings, norms, k: int = 3) -> list[dict]:
    q_emb  = compute_embeddings([query])[0]
    q_norm = np.linalg.norm(q_emb)
    scores = [
        np.dot(q_emb, e) / (q_norm * n + 1e-8)
        for e, n in zip(embeddings, norms)
    ]
    top_idxs = np.argsort(scores)[-k:][::-1]
    return [records[i] for i in top_idxs]

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="📰 Chatbot Interaktif Harian Kompas", layout="centered")
st.logo(
    "http://satrio.pw/wp-content/uploads/2025/05/Logo-Ikon-Kompas-85x85-1.png",  # gambar utama
    link="https://www.kompas.id",  # ke mana logo akan mengarah kalau diklik
    icon_image="http://satrio.pw/wp-content/uploads/2025/05/Logo-Ikon-Kompas-85x85-1.png",  # ikon kecil di title‑bar (optional)
)
st.image("http://satrio.pw/wp-content/uploads/2025/05/Logo-Ikon-Kompas-45x45-1.png")
st.badge(":warning: Versi Pre-alpha", color="orange")
st.title("Chatbot Interaktif Harian Kompas")
st.write("Chatbot untuk membantu menjawab pertanyaan pembaca Kompas tentang berita/reportase secara interaktif.")


# Load and vectorize knowledge
records = load_knowledge()
if not records:
    st.stop()
embeddings, norms = build_vector_store(records)

# Render chat history
if "history" not in st.session_state:
    st.session_state.history = []
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Tanyakan sesuatu tentang berita kami…")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    # Retrieve relevant paragraphs
    related = retrieve(user_input, records, embeddings, norms)
    context = "\n\n---\n\n".join(f"**{r['title']}**\n{r['content']}" for r in related)

    system_prompt = (
        "Anda adalah asisten yang hanya menjawab berdasarkan basis pengetahuan berikut:\n\n"
        f"{context}\n\n"
        "Jika pertanyaan di luar cakupan, jawab: "
        "'Maaf, saya tidak memiliki informasi tersebut.'"
        "Narasumber ekonom, pakar, ahli, pengajar, pengamat, praktisi juga disebut pakar."
        "di akhir respon, berikan judul berita yang memuat url atau link sumber respons."
    )

    with st.spinner("Memproses…"):
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_input},
            ],
        )
        answer = resp.choices[0].message.content.strip()

    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
