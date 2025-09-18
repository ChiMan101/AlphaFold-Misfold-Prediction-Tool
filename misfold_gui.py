# misfold_gui.py
# -*- coding: utf-8 -*-
import os
import re
import io
import csv
import gzip
import time
import json
import math
import sqlite3
import hashlib
import secrets
import requests
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from Bio.PDB import PDBParser

# =========================================================
# Streamlit page config
# =========================================================
st.set_page_config(page_title="AlphaFold Misfold Prediction Tool", layout="wide")

# =========================================================
# THEME: light/dark toggle (CSS variables)
# =========================================================
def apply_theme(theme: str):
    """
    Inject CSS variables and component styles for light/dark modes.
    """
    if theme not in ("dark", "light"):
        theme = "dark"

    if theme == "dark":
        css = """
        <style>
          :root{
            --bg:#0f1116; --fg:#ffffff; --muted:#b9c0cc; --card:#171a22;
            --border:#262b36; --accent:#69a3da; --accent-2:#8ec0ff;
          }
        </style>
        """
    else:
        css = """
        <style>
          :root{
            --bg:#ffffff; --fg:#111416; --muted:#4a4f57; --card:#f6f7f9;
            --border:#e2e6ee; --accent:#225caa; --accent-2:#427fd8;
          }
        </style>
        """

    base = """
    <style>
      html, body, .stApp { background: var(--bg) !important; color: var(--fg) !important; }
      .stMarkdown, p, h1,h2,h3,h4,h5,h6, label, span, div { color: var(--fg); }
      /* Cards/containers */
      .stTabs [data-baseweb="tab"] { color: var(--fg); }
      .stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }
      .stTextInput, .stTextArea, .stDownloadButton, .stFileUploader { color: var(--fg); }
      /* Sidebar */
      section[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
      /* Header bar */
      .topbar{
          position:sticky; top:0; z-index:999; background:var(--card); padding:10px 16px;
          border-bottom:1px solid var(--border); display:flex; align-items:center; gap:16px;
      }
      /* Footer bar */
      .footer-bar{
          position:fixed;left:0;bottom:0;width:100%;
          background:var(--card);color:var(--fg);padding:8px 12px;
          font-size:14px;z-index:100;border-top:1px solid var(--border);
          display:flex;gap:10px;justify-content:center;align-items:center;
      }
      .footer-bar a{color:var(--accent);text-decoration:none;font-weight:600;}
      .footer-bar a:hover{text-decoration:underline;}
      /* Buttons */
      button[kind="secondary"], .stButton>button {
          background: var(--accent); color: white; border: 0; border-radius: 8px;
      }
      .stButton>button:hover { background: var(--accent-2); }
      /* Code blocks */
      pre, code { background: var(--card) !important; color: var(--fg) !important; }
      /* Tables */
      .stDataFrame { border: 1px solid var(--border); border-radius: 8px; }
    </style>
    """
    st.markdown(css + base, unsafe_allow_html=True)


# =========================================================
# Auth (SQLite + PBKDF2)
# =========================================================
AUTH_DB = "auth.db"

def _auth_conn():
    c = sqlite3.connect(AUTH_DB)
    c.row_factory = sqlite3.Row
    return c

def init_auth_db():
    with _auth_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150_000).hex()

def create_user(username: str, password: str) -> bool:
    username = username.strip()
    if not username or not password:
        return False
    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    try:
        with _auth_conn() as con:
            con.execute(
                "INSERT INTO users (username, hash, salt, created_at) VALUES (?, ?, ?, datetime('now'))",
                (username, pwd_hash, salt),
            )
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str) -> bool:
    with _auth_conn() as con:
        cur = con.execute("SELECT hash, salt FROM users WHERE username = ?", (username.strip(),))
        row = cur.fetchone()
        if not row:
            return False
        calc = _hash_password(password, row["salt"])
        return secrets.compare_digest(calc, row["hash"])

init_auth_db()

def is_logged_in() -> bool:
    return bool(st.session_state.get("user"))

def current_user() -> str | None:
    return st.session_state.get("user")

def login(username: str):
    st.session_state["user"] = username

def logout():
    st.session_state.pop("user", None)

# =========================================================
# App data (SQLite): Messenger + History
# =========================================================
APP_DB = "misfold_app.db"

def _app_conn():
    c = sqlite3.connect(APP_DB)
    c.row_factory = sqlite3.Row
    return c

def init_app_db():
    with _app_conn() as con:
        # Messenger
        con.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_key TEXT NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_key)")
        # History
        con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            filename TEXT NOT NULL,
            accession TEXT,
            error REAL,
            threshold REAL,
            misfold INTEGER,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON history(username)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_accession ON history(accession)")

init_app_db()

def add_msg(thread_key: str, author: str, body: str):
    if not body or not body.strip():
        return
    author = (author or "Analyst").strip()
    with _app_conn() as con:
        con.execute(
            "INSERT INTO messages (thread_key, author, body, created_at) VALUES (?, ?, ?, datetime('now'))",
            (thread_key, author, body.strip()),
        )

def list_msgs(thread_key: str, limit: int = 200):
    with _app_conn() as con:
        cur = con.execute(
            "SELECT author, body, created_at FROM messages WHERE thread_key = ? ORDER BY id DESC LIMIT ?",
            (thread_key, limit),
        )
        return list(cur.fetchall())

def add_history(username: str | None, filename: str, accession: str | None,
                error: float | None, threshold: float | None, misfold: int):
    with _app_conn() as con:
        con.execute(
            "INSERT INTO history (username, filename, accession, error, threshold, misfold, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (username, filename, accession, error, threshold, misfold),
        )

def list_history(username: str | None, limit: int = 200):
    with _app_conn() as con:
        if username:
            cur = con.execute("SELECT * FROM history WHERE username = ? ORDER BY id DESC LIMIT ?", (username, limit))
        else:
            cur = con.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,))
        return list(cur.fetchall())

def clear_history(username: str):
    with _app_conn() as con:
        con.execute("DELETE FROM history WHERE username = ?", (username,))

# =========================================================
# Model & inference (standalone AE)
# =========================================================
class FoldAutoencoder(nn.Module):
    def __init__(self, input_dim=300, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

@st.cache_resource
def load_model(model_path="trained_autoencoder.pt"):
    model = FoldAutoencoder()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def extract_plddt_scores(file_path: str, max_len: int = 300) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as fh:
            structure = parser.get_structure("protein", fh)
    else:
        structure = parser.get_structure("protein", file_path)
    scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    scores.append(residue["CA"].bfactor)  # AlphaFold stores pLDDT here
    scores = scores[:max_len]
    if len(scores) < max_len:
        scores += [0.0] * (max_len - len(scores))
    return np.array(scores, dtype=np.float32)

def infer_misfold(model: FoldAutoencoder, raw_scores: np.ndarray, threshold: float = 0.03):
    x = (raw_scores / 100.0).astype(np.float32)             # normalize to 0–1
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x_tensor)
        error = torch.mean((out - x_tensor) ** 2).item()    # per-residue MSE
    return (error > threshold), error

# =========================================================
# Text helpers (PDB header & UniProt)
# =========================================================
def get_protein_description(file_path: str) -> str:
    opener = gzip.open if file_path.endswith(".gz") else open
    lines = []
    with opener(file_path, "rt") as fh:
        for line in fh:
            if line.startswith(("HEADER", "TITLE", "COMPND")):
                lines.append(line.rstrip())
            if len(lines) >= 20:
                break
    return "\n".join(lines) if lines else "No description found in PDB file."

def parse_uniprot_from_text(text: str):
    m = re.search(r'\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b', text)
    if m: return m.group(0)
    m10 = re.search(r'\b[A-NR-Z0-9]{1}[0-9][A-Z0-9]{3}[0-9][A-Z0-9]{4}\b', text)
    return m10.group(0) if m10 else None

def extract_uniprot_accession(file_path: str):
    fname = os.path.basename(file_path)
    m = re.search(r'AF-([A-Z0-9]+)-F\d+', fname)
    if m: return m.group(1)
    acc = parse_uniprot_from_text(fname)
    if acc: return acc
    opener = gzip.open if file_path.endswith(".gz") else open
    text = []
    with opener(file_path, "rt") as fh:
        for i, line in enumerate(fh):
            if line.startswith(("HEADER", "TITLE", "COMPND", "DBREF", "REMARK")):
                text.append(line)
            if i > 500:
                break
    return parse_uniprot_from_text(" ".join(text)) if text else None

@st.cache_data(show_spinner=False)
def fetch_uniprot_function(accession: str):
    if not accession:
        return None, None, None
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None, None, f"https://www.uniprot.org/uniprotkb/{accession}"
        data = r.json()
        try:
            name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
        except Exception:
            name = data.get("primaryAccession", accession)
        function_text = None
        for c in data.get("comments", []):
            if c.get("commentType") == "FUNCTION":
                tx = c.get("texts", [])
                if tx:
                    function_text = tx[0].get("value"); break
        if not function_text:
            go_terms = []
            for dbref in data.get("uniProtKBCrossReferences", []):
                if dbref.get("database") == "GO":
                    props = {p["key"]: p["value"] for p in dbref.get("properties", [])}
                    if props.get("aspect") == "F":
                        go_terms.append(props.get("term"))
            if go_terms:
                function_text = "GO molecular function terms: " + "; ".join(go_terms)
        return function_text, name, f"https://www.uniprot.org/uniprotkb/{accession}"
    except Exception:
        return None, None, f"https://www.uniprot.org/uniprotkb/{accession}"

# =========================================================
# Chatbot helpers (context + lightweight QA)
# =========================================================
def summarize_plddt(scores: np.ndarray) -> Dict[str, Any]:
    if scores is None or len(scores) == 0:
        return {"n": 0, "mean": None, "min": None, "max": None, "low_segments": []}
    s = np.array(scores, dtype=float)
    n = len(s)
    mean = float(np.mean(s))
    minv = float(np.min(s))
    maxv = float(np.max(s))
    low_segments: List[Tuple[int, int]] = []
    in_seg = False; start = 0
    for i, v in enumerate(s):
        if v < 50 and not in_seg:
            in_seg = True; start = i
        if (v >= 50 or i == n - 1) and in_seg:
            end = i if v >= 50 else i
            low_segments.append((start + 1, end + 1))
            in_seg = False
    return {"n": n, "mean": mean, "min": minv, "max": maxv, "low_segments": low_segments}

def build_protein_context(filename, accession, uni_name, uni_function, header_text, scores, recon_error, threshold) -> str:
    stats = summarize_plddt(scores if scores is not None else np.array([]))
    segs = ", ".join([f"{a}-{b}" for a, b in stats["low_segments"]]) if stats["low_segments"] else "None detected (<50)"
    ctx = [
        f"FILE: {filename}",
        f"UNIPROT ACCESSION: {accession or 'unknown'}",
        f"PROTEIN NAME: {uni_name or 'unknown'}",
        f"FUNCTION (UniProt): {uni_function or 'not available'}",
        f"pLDDT (per-residue confidence): N={stats['n']}, mean={stats['mean']:.2f} min={stats['min']:.2f} max={stats['max']:.2f}" if stats['n'] else "pLDDT: no data",
        f"Low-confidence segments (pLDDT<50): {segs}",
        f"Autoencoder reconstruction error: {recon_error:.4f}" if recon_error is not None else "Reconstruction error: n/a",
        f"Threshold used: {threshold:.4f}" if threshold is not None else "Threshold: n/a",
        "HEADER (PDB): " + (header_text.strip().replace("\n", " ")[:800] + ("..." if header_text and len(header_text) > 800 else "")) if header_text else "HEADER: n/a",
    ]
    return "\n".join(ctx)

def local_answer(question: str, ctx: str) -> str | None:
    q = question.lower()
    if any(k in q for k in ["name", "what protein", "protein name"]):
        for line in ctx.splitlines():
            if line.startswith("PROTEIN NAME:"): return line.replace("PROTEIN NAME:", "").strip() or "Name not available."
    if "accession" in q or "uniprot" in q:
        for line in ctx.splitlines():
            if line.startswith("UNIPROT ACCESSION:"): return line.replace("UNIPROT ACCESSION:", "").strip()
    if "function" in q or "what does it do" in q:
        for line in ctx.splitlines():
            if line.startswith("FUNCTION (UniProt):"):
                ans = line.split(":", 1)[1].strip()
                return ans if ans and ans != "not available" else "Function not available from UniProt."
    if any(k in q for k in ["plddt", "confidence", "quality"]):
        for line in ctx.splitlines():
            if line.startswith("pLDDT"): return line
    if "low" in q and ("segment" in q or "region" in q or "disorder" in q):
        for line in ctx.splitlines():
            if line.startswith("Low-confidence segments"): return line
    if any(k in q for k in ["misfold", "mis-fold", "fold", "anomal"]):
        lines = [l for l in ctx.splitlines() if l.startswith(("Autoencoder reconstruction error", "Threshold used", "Low-confidence segments"))]
        return " • ".join(lines) if lines else "No inference stats available."
    if "header" in q:
        return "\n".join([l for l in ctx.splitlines() if l.startswith("HEADER")]) or "No PDB header available."
    return None

def call_openai_chat(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.2, max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

# =========================================================
# Header / Footer
# =========================================================
def render_header():
    st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)
    cols = st.columns([4,2.2,2.2,2.6,2.0])
    with cols[0]:
        st.markdown("### 🧬 AlphaFold Misfold Prediction Tool")
    if not is_logged_in():
        with cols[1]:
            user = st.text_input("Username", key="login_username", label_visibility="collapsed", placeholder="Username")
        with cols[2]:
            pwd = st.text_input("Password", type="password", key="login_password", label_visibility="collapsed", placeholder="Password")
        with cols[3]:
            c1, c2 = st.columns(2)
            if c1.button("Login", use_container_width=True):
                if verify_user(user, pwd):
                    login(user); st.success(f"Welcome, {user}!"); st.rerun()
                else:
                    st.error("Invalid username or password.")
            if c2.button("Sign up", use_container_width=True):
                if not user or not pwd:
                    st.warning("Enter a username and password to sign up.")
                else:
                    ok = create_user(user, pwd)
                    st.success("Account created. You can now log in.") if ok else st.error("Username already exists.")
    else:
        with cols[1]:
            st.markdown(f"**Signed in as:** `{current_user()}`")
        with cols[4]:
            if st.button("Logout", use_container_width=True):
                logout(); st.success("Logged out."); st.rerun()

def render_footer():
    st.markdown(
        """
        <div class="footer-bar">
            🧬 AlphaFold Misfold Prediction Tool • <a href="#messenger">Open Messenger 💬</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# Messenger UI
# =========================================================
def render_messenger(thread_key: str, default_body: str = ""):
    st.markdown('<a name="messenger"></a>', unsafe_allow_html=True)
    st.subheader("💬 Messenger (this protein)")
    st.caption(f"Thread key: `{thread_key}` — messages persist in a local SQLite DB.")

    if not is_logged_in():
        st.info("Please **log in** (top bar) to post messages.")
        st.text_area("Message (login required to send)", value=default_body, height=100, disabled=True)
        st.button("Send", disabled=True)
    else:
        author = current_user() or "Analyst"
        body = st.text_area("Message", value=default_body, height=100, key=f"compose_{thread_key}")
        c1, c2, _ = st.columns([1,1,6])
        with c1:
            if st.button("Send", key=f"send_{thread_key}"):
                add_msg(thread_key, author, body); st.success("Sent."); st.rerun()
        with c2:
            if st.button("Refresh", key=f"refresh_{thread_key}"):
                st.rerun()

    msgs = list_msgs(thread_key)
    if not msgs:
        st.info("No messages yet. Be the first to comment on this protein.")
    else:
        for m in msgs:
            with st.container():
                st.markdown(f"**{m['author']}** · {m['created_at']}")
                st.write(m["body"])
                st.markdown("---")

# =========================================================
# Sidebar (controls + THEME)
# =========================================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Controls")
        # Theme selector
        theme = st.radio("Theme", options=["Dark","Light"], index=0 if st.session_state.get("theme","dark")=="dark" else 1, horizontal=True)
        theme_val = "dark" if theme == "Dark" else "light"
        st.session_state["theme"] = theme_val
        apply_theme(theme_val)

        uploaded = st.file_uploader("Upload .pdb or .pdb.gz", type=["pdb", "gz"])

        st.markdown("**Model**")
        model_path = st.text_input("Model path", value="trained_autoencoder.pt",
                                   help="Path to your saved PyTorch model (.pt)")
        model_exists = os.path.exists(model_path)
        st.caption(("✅ Found model" if model_exists else "❌ Model not found") + f": `{model_path}`")

        st.markdown("**Prediction Settings**")
        threshold = st.slider("Misfold threshold (reconstruction MSE on normalized pLDDT)",
                              0.001, 0.2, 0.03, 0.001)

        st.markdown("**Display Options**")
        show_uniprot = st.checkbox("Fetch UniProt function", value=True)
        show_raw_header = st.checkbox("Show raw PDB header lines", value=False)

        st.markdown("---")
        with st.expander("About"):
            st.write(
                "Upload an AlphaFold structure (.pdb or .pdb.gz) to visualize per-residue pLDDT, "
                "fetch a UniProt function summary, and estimate potential misfolding via an autoencoder trained on pLDDT profiles."
            )
            st.write("• **pLDDT** is AlphaFold’s per-residue confidence score (0–100). Low pLDDT often indicates flexible/disordered regions.")
            st.markdown("### 🔬 What are PDB files?")
            st.write(
                "A **PDB file** (Protein Data Bank) stores the 3D coordinates of biomolecules (proteins, DNA, RNA), "
                "along with metadata. AlphaFold’s PDBs include predicted structures and store **pLDDT** scores in the B-factor column. "
                "Uploading a PDB lets this app extract pLDDT, look up the protein’s function from UniProt, and assess potential misfolding."
            )

        return uploaded, model_path, model_exists, threshold, show_uniprot, show_raw_header

# =========================================================
# History UI
# =========================================================
def render_history_tab():
    st.subheader("🗂 History")
    user = current_user()
    rows = list_history(user, limit=500) if user else list_history(None, limit=200)
    st.caption("Showing **your** recent analyses." if user else "Showing **recent public** analyses (not logged in).")

    if not rows:
        st.info("No history yet."); return

    import pandas as pd
    df = pd.DataFrame(rows, columns=rows[0].keys())
    df["misfold"] = df["misfold"].map({0: "No", 1: "Yes"})
    st.dataframe(df, use_container_width=True)

    buf = io.StringIO(); writer = csv.writer(buf)
    writer.writerow(df.columns)
    for _, rec in df.iterrows():
        writer.writerow([rec.get(c, "") for c in df.columns])
    st.download_button("⬇️ Download CSV", data=buf.getvalue(), file_name="misfold_history.csv", mime="text/csv")

    if user and st.button("🧹 Clear my history"):
        clear_history(user); st.success("Cleared your history."); st.rerun()

# =========================================================
# Chatbot tab
# =========================================================
def render_chatbot_tab(filename, accession, uni_name, uni_function, header_text, scores, recon_error, threshold):
    st.subheader("🤖 Chatbot")
    ctx = build_protein_context(filename, accession, uni_name, uni_function, header_text, scores, recon_error, threshold)
    with st.expander("View protein context used by the chatbot"):
        st.code(ctx, language="text")

    chat_key = f"chat_{accession or filename}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for turn in st.session_state[chat_key]:
        role = "You" if turn["role"] == "user" else "Assistant"
        st.markdown(f"**{role}:** {turn['content']}")

    q = st.text_input("Ask a question about this protein", placeholder="e.g., Which regions have low confidence? What does this protein do?")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Ask"):
            if q.strip():
                st.session_state[chat_key].append({"role": "user", "content": q.strip()})
                local = local_answer(q, ctx)
                if local:
                    st.session_state[chat_key].append({"role": "assistant", "content": local}); st.rerun()
                try:
                    sys_prompt = ("You are a helpful protein analysis assistant. Answer concisely using ONLY the provided context. "
                                  "If the answer is not in context, say you don't have enough information. Do not hallucinate.\n\nContext:\n" + ctx)
                    ans = call_openai_chat(sys_prompt, q.strip())
                    st.session_state[chat_key].append({"role": "assistant", "content": ans})
                except Exception:
                    st.session_state[chat_key].append({"role":"assistant","content":
                        "I don't have enough information beyond the context above. "
                        "Try asking about: pLDDT stats, low-confidence regions, UniProt function, accession, or misfolding result."
                    })
                st.rerun()
    with col2:
        if st.button("Clear chat"):
            st.session_state.pop(chat_key, None); st.success("Cleared."); st.rerun()

# =========================================================
# Main App
# =========================================================
def main():
    # Apply theme early (default to dark)
    apply_theme(st.session_state.get("theme", "dark"))

    render_header()
    uploaded, model_path, model_exists, threshold, show_uniprot, show_raw_header = render_sidebar()

    st.title("Results • Messenger • History • Chatbot")

    tab_results, tab_chat, tab_hist, tab_bot = st.tabs(["Results", "Messenger", "History", "Chatbot"])

    # ===== RESULTS TAB =====
    with tab_results:
        if uploaded is None:
            st.info("Upload a `.pdb` or `.pdb.gz` file in the sidebar to begin.")
            accession = None; description = None; uni_name = None; function_text = None; scores = None; error = None
        else:
            tmp_path = f"temp_{uploaded.name}"
            with open(tmp_path, "wb") as f: f.write(uploaded.read())

            accession = None; description = ""
            try:
                description = get_protein_description(tmp_path)
                accession = extract_uniprot_accession(tmp_path)
            except Exception as e:
                st.warning(f"Could not parse header/accession: {e}")

            st.subheader("📄 Protein Description")
            if description and description.strip():
                if show_raw_header: st.code(description, language="text")
                else:
                    lines = description.splitlines()
                    st.write("\n".join(lines[:5] if len(lines) > 5 else lines))
                    if len(lines) > 5: st.caption("…toggle **Show raw PDB header lines** in the sidebar to view all header lines.")
            else:
                st.info("No PDB header description found.")

            uni_name = None; function_text = None
            if show_uniprot:
                st.subheader("🧾 UniProt Function")
                if accession:
                    function_text, uni_name, uni_url = fetch_uniprot_function(accession)
                    header = f"**{uni_name or accession}**  •  Accession: `{accession}`"
                    if uni_url: header += f"  •  [View on UniProt]({uni_url})"
                    st.markdown(header)
                    st.write(function_text) if function_text else st.info("No function text available from UniProt for this accession.")
                else:
                    st.warning("Couldn’t determine a UniProt accession from the filename/header.")

            st.subheader("📊 pLDDT Score Profile")
            try:
                scores = extract_plddt_scores(tmp_path)  # raw 0–100 for the plot
                st.line_chart(scores, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to extract pLDDT scores: {e}"); scores = None

            st.subheader("🔍 Prediction Result")
            error = None; is_mf_flag = None
            if not model_exists:
                st.error(f"Model not found at `{model_path}`. Provide a valid path in the sidebar.")
            elif scores is None:
                st.error("No scores to evaluate.")
            else:
                try:
                    session_key = "_ae_model_" + model_path
                    model = st.session_state.get(session_key) or load_model(model_path)
                    st.session_state[session_key] = model
                    is_misfold, error = infer_misfold(model, scores, threshold=threshold)
                    is_mf_flag = 1 if is_misfold else 0
                    st.markdown(f"**Reconstruction Error**: `{error:.4f}`  •  **Threshold**: `{threshold:.3f}`")
                    st.error("⚠️ Likely Misfolded (error > threshold)") if is_misfold else st.success("✅ Structure appears typical (error ≤ threshold)")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            try:
                if scores is not None and error is not None and is_mf_flag is not None:
                    add_history(current_user(), uploaded.name, accession, float(error), float(threshold), int(is_mf_flag))
            except Exception as e:
                st.warning(f"Could not record history: {e}")

            try: os.remove(tmp_path)
            except Exception: pass

    # ===== MESSENGER TAB =====
    with tab_chat:
        rows = list_history(current_user(), limit=1) if is_logged_in() else list_history(None, limit=1)
        thread_key = (rows[0]["accession"] or rows[0]["filename"]) if rows else "general"
        render_messenger(thread_key, default_body="")
        st.markdown('<a name="messenger"></a>', unsafe_allow_html=True)

    # ===== HISTORY TAB =====
    with tab_hist:
        render_history_tab()

    # ===== CHATBOT TAB =====
    with tab_bot:
        filename = uploaded.name if uploaded is not None else "no_file"
        acc = accession if "accession" in locals() else None
        u_name = uni_name if "uni_name" in locals() else None
        u_func = function_text if "function_text" in locals() else None
        header = description if "description" in locals() else None
        sc = scores if "scores" in locals() else None
        err = error if "error" in locals() else None
        thr = threshold if "threshold" in locals() else None

        render_chatbot_tab(filename, acc, u_name, u_func, header, sc, err, thr)

    render_footer()

if __name__ == "__main__":
    main()
