# utils/rag_engine.py

import os
import json
import hashlib
import numpy as np

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def _plugin_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def rag_doc_dir():
    return os.path.join(_plugin_root(), "rag_doc")

def _index_dir():
    d = os.path.join(_plugin_root(), ".rag_index")
    os.makedirs(d, exist_ok=True)
    return d

def list_rag_txt_files():
    root = rag_doc_dir()
    if not os.path.isdir(root):
        return []
    return [f for f in os.listdir(root) if f.lower().endswith(".txt")]

def _read_text(fp):
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _chunk_text(text, chunk_size=800, overlap=100):
    # 단순 문자 길이 기반 청킹
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def _hash_for_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _embed_sentences(sentences):
    """
    1순위: sentence-transformers
    2순위: scikit-learn TfidfVectorizer
    3순위: 간단한 백오프(문자 unigram 빈도)
    """
    # 1) sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_EMBED_MODEL_NAME, device="cpu")  # CPU에도 충분히 작음
        emb = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return emb
    except Exception:
        pass

    # 2) scikit-learn TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(max_features=4096)
        X = vect.fit_transform(sentences).astype(np.float32)
        # L2 normalize
        row_norm = np.sqrt((X.power(2)).sum(axis=1)).A1 + 1e-12
        X = X.multiply(1.0 / row_norm[:, None]).A
        return X
    except Exception:
        pass

    # 3) 아주 간단한 백오프: a-z 빈도(26차원)
    def char_feat(s):
        s = s.lower()
        vec = np.zeros(26, dtype=np.float32)
        for ch in s:
            if "a" <= ch <= "z":
                vec[ord(ch) - 97] += 1.0
        if np.linalg.norm(vec) > 0:
            vec /= np.linalg.norm(vec)
        return vec
    return np.stack([char_feat(s) for s in sentences], axis=0)


def build_or_load_index(txt_filename):
    """
    반환: dict {
        "chunks": [str, ...],
        "emb": np.ndarray (n_chunks, d),
        "txt_path": fullpath
    }
    """
    src = os.path.join(rag_doc_dir(), txt_filename)
    if not os.path.exists(src):
        raise FileNotFoundError(f"RAG txt not found: {src}")

    sig = _hash_for_file(src)
    cache_fp = os.path.join(_index_dir(), f"{os.path.basename(src)}.{sig}.npz")

    if os.path.exists(cache_fp):
        data = np.load(cache_fp, allow_pickle=True)
        return {"chunks": data["chunks"].tolist(), "emb": data["emb"].astype(np.float32), "txt_path": src}

    text = _read_text(src)
    chunks = _chunk_text(text, chunk_size=800, overlap=100)
    emb = _embed_sentences(chunks).astype(np.float32)

    np.savez_compressed(cache_fp, chunks=np.array(chunks, dtype=object), emb=emb)
    return {"chunks": chunks, "emb": emb, "txt_path": src}

def _cosine_topk(mat, q, k=4):
    # mat: (n, d), q: (d,)
    s = mat @ q  # 가정: 이미 L2 정규화
    idx = np.argsort(-s)[:k]
    return idx, s[idx]

def query_with_rag(index, query, k=4):
    emb_q = _embed_sentences([query])[0].astype(np.float32)
    idx, _ = _cosine_topk(index["emb"], emb_q, k=k)
    picks = [index["chunks"][i] for i in idx]
    return "\n---\n".join(picks)
