import os
import re
import uuid
from typing import List, Tuple, Dict, Any

import chainlit as cl
from groq import Groq

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 20
COSINE_DISTANCE_THRESHOLD = 0.55
MAX_HITS_AFTER_GATE = 8
MAX_CONTEXT_CHARS = 8000


def read_txt_or_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_pages(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        out.append((i + 1, page.extract_text() or ""))
    return out


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def make_chunks_with_metadata(
    *,
    filename: str,
    ext: str,
    path: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    chunks: List[str] = []
    metas: List[Dict[str, Any]] = []

    if ext in (".txt", ".md"):
        text = read_txt_or_md(path)
        cks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for j, c in enumerate(cks):
            chunks.append(c)
            metas.append({"filename": str(filename), "chunk_index": int(j)})

    elif ext == ".pdf":
        pages = read_pdf_pages(path)
        idx = 0
        for page_num, page_text in pages:
            cks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
            for c in cks:
                chunks.append(c)
                metas.append({"filename": str(filename), "page": int(page_num), "chunk_index": int(idx)})
                idx += 1
    else:
        raise ValueError("Formato non supportato (solo txt/md/pdf).")

    return chunks, metas


def get_embedder() -> SentenceTransformer:
    embedder = cl.user_session.get("embedder")
    if embedder is None:
        embedder = SentenceTransformer(_EMBED_MODEL_NAME)
        cl.user_session.set("embedder", embedder)
    return embedder


def get_chroma_collection() -> chromadb.api.models.Collection.Collection:
    collection_name = cl.user_session.get("collection_name")
    if not collection_name:
        collection_name = f"kb_{uuid.uuid4().hex}"
        cl.user_session.set("collection_name", collection_name)

    client = cl.user_session.get("chroma_client")
    if client is None:
        client = chromadb.PersistentClient(
            path=".chroma_db",
            settings=Settings(anonymized_telemetry=False),
        )
        cl.user_session.set("chroma_client", client)

    try:
        return client.get_collection(collection_name)
    except Exception:
        try:
            return client.create_collection(
                name=collection_name,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception:
            return client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )


def index_chunks(chunks: List[str], metadatas: List[Dict[str, Any]]) -> int:
    clean = [(c, m) for c, m in zip(chunks, metadatas) if (c or "").strip()]
    if not clean:
        return 0
    chunks2, metas2 = zip(*clean)
    chunks2, metas2 = list(chunks2), list(metas2)

    embedder = get_embedder()
    col = get_chroma_collection()
    embeddings = embedder.encode(chunks2, normalize_embeddings=True).tolist()
    ids = [f"chunk_{uuid.uuid4().hex}" for _ in chunks2]
    col.add(ids=ids, documents=chunks2, metadatas=metas2, embeddings=embeddings)
    return len(chunks2)


def retrieve(query: str, k: int = TOP_K) -> List[Tuple[str, Dict[str, Any], float]]:
    embedder = get_embedder()
    col = get_chroma_collection()
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: List[Tuple[str, Dict[str, Any], float]] = []
    for d, m, dist in zip(docs, metas, dists):
        out.append((d or "", m or {}, float(dist)))
    return out


def quality_gate(hits: List[Tuple[str, Dict[str, Any], float]]) -> List[Tuple[str, Dict[str, Any], float]]:
    if not hits:
        return []
    good = [(d, m, dist) for (d, m, dist) in hits if dist <= COSINE_DISTANCE_THRESHOLD]
    if not good:
        return hits[: min(3, len(hits))]
    return good[:MAX_HITS_AFTER_GATE]


def select_context(hits: List[Tuple[str, Dict[str, Any], float]], max_chars: int = MAX_CONTEXT_CHARS) -> List[str]:
    context_chunks: List[str] = []
    used = 0
    for doc, meta, dist in hits:
        if used >= max_chars:
            break
        filename = meta.get("filename", "documento")
        page = meta.get("page", None)
        chunk_index = meta.get("chunk_index", None)

        where = f"{filename}"
        if isinstance(page, int):
            where += f" | pag. {page}"
        if isinstance(chunk_index, int):
            where += f" | chunk {chunk_index}"

        header = f"[FONTE: {where} | dist={dist:.3f}]\n"
        piece = header + (doc or "").strip()

        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(piece) > remaining:
            piece = piece[:remaining]

        context_chunks.append(piece)
        used += len(piece)
    return context_chunks


def groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Manca GROQ_API_KEY.")
    return Groq(api_key=key)


def answer_with_groq(question: str, context_chunks: List[str]) -> str:
    client = groq_client()
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
    system = (
        "Rispondi usando solo il contenuto fornito. "
        "Se non ci sono informazioni sufficienti nel contenuto, dillo chiaramente. "
        "Quando possibile, cita brevemente un passaggio e la sua FONTE."
    )
    user = f"CONTENUTO:\n{context}\n\nDOMANDA:\n{question}\n"
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


@cl.on_chat_start
async def start():
    await cl.Message(content="Carica un file txt / md / pdf.").send()

    files = await cl.AskFileMessage(
        content="Carica un file:",
        accept=["text/plain", "text/markdown", "application/pdf"],
        max_size_mb=50,
        timeout=180,
    ).send()

    if not files:
        await cl.Message(content="Nessun file ricevuto.").send()
        return

    f = files[0]
    path = f.path
    name = f.name or os.path.basename(path)
    ext = os.path.splitext(name.lower())[1]

    try:
        chunks, metas = make_chunks_with_metadata(filename=name, ext=ext, path=path)
        cl.user_session.set("collection_name", f"kb_{uuid.uuid4().hex}")
        n = index_chunks(chunks, metas)
        if n <= 0:
            await cl.Message(content="File vuoto o non leggibile.").send()
            return
        cl.user_session.set("has_kb", True)
        await cl.Message(content=f"File {name} indicizzato. Chunk: {n}.").send()
    except Exception as e:
        await cl.Message(content=f"Errore: {e}").send()


@cl.on_message
async def main(message: cl.Message):
    if not cl.user_session.get("has_kb"):
        await cl.Message(content="Carica prima un file.").send()
        return

    question = (message.content or "").strip()
    if not question:
        await cl.Message(content="Scrivi una domanda.").send()
        return

    try:
        hits = retrieve(question, k=TOP_K)
        hits = quality_gate(hits)
        if not hits:
            await cl.Message(content="Non trovo informazioni nel documento per rispondere.").send()
            return
        context_chunks = select_context(hits, max_chars=MAX_CONTEXT_CHARS)
        response = answer_with_groq(question, context_chunks)
        await cl.Message(content=response).send()
    except Exception as e:
        await cl.Message(content=f"Errore: {e}").send()