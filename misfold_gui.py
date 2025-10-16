# misfold_gui.py
# -*- coding: utf-8 -*-
import os
import re
import io
import csv
import gzip
import json
import sqlite3
import hashlib
import secrets
import requests
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from Bio.PDB import PDBParser
from dotenv import load_dotenv
load_dotenv()  # loads variables from .env into os.environ


# =========================================================
# Streamlit page config
# =========================================================
st.set_page_config(page_title="AlphaFold Misfold Prediction Tool", layout="wide")

# =========================================================
# THEME: Dark-only with Open Sans + ChatGPT-like layout CSS
# =========================================================
def apply_theme(_: str = "dark"):
    css_vars = """
    <style>
      :root{
        --bg:#0f1116; --fg:#ffffff; --muted:#b9c0cc; --card:#171a22;
        --border:#262b36; --accent:#69a3da; --accent-2:#8ec0ff;
        --danger:#d9534f; --success:#38a169;
      }
    </style>
    """
    base = """
    <style>
      /* Load Open Sans (400/600/700) */
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

      /* Global font + colors */
      html, body, .stApp {
        background: var(--bg) !important;
        color: var(--fg) !important;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                     Roboto, 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 400;
      }

      .stMarkdown, p, h1,h2,h3,h4,h5,h6, label, span, div { color: var(--fg); }
      section[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
      .topbar{
          position:sticky; top:0; z-index:999; background:var(--card); padding:8px 12px;
          border-bottom:1px solid var(--border); display:flex; align-items:center; gap:16px;
      }
      .footer-bar{
          position:fixed;left:0;bottom:0;width:100%;
          background: var(--card); color: var(--fg); padding:8px 12px;
          font-size:14px; z-index:100; border-top:1px solid var(--border);
          display:flex; gap:14px; justify-content:center; align-items:center; flex-wrap:wrap;
      }
      .footer-bar a{color:var(--accent); text-decoration:none; font-weight:700;}
      .footer-bar a.active{color:var(--accent-2); text-decoration:underline;}
      .footer-bar a:hover{text-decoration:underline;}

      /* Buttons */
      .stButton > button,
      .stDownloadButton > button,
      button[kind="primary"] {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        padding: 0.55rem 0.95rem !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.03s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.18) !important;
        font-family: 'Open Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important;
      }
      .stButton > button:hover,
      .stDownloadButton > button:hover,
      button[kind="primary"]:hover { background: var(--accent-2) !important; border-color: var(--accent-2) !important; }
      .stButton > button:active,
      .stDownloadButton > button:active,
      button[kind="primary"]:active { transform: translateY(1px); }
      .stButton > button:focus-visible,
      .stDownloadButton > button:focus-visible,
      button[kind="primary"]:focus-visible {
        outline: 3px solid rgba(142,192,255,0.35) !important; outline-offset: 2px !important;
      }

      /* Secondary / ghost */
      button[kind="secondary"] {
        background: transparent !important; color: var(--fg) !important;
        border: 1px solid var(--border) !important; border-radius: 8px !important;
        font-family: 'Open Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 600 !important;
      }
      button[kind="secondary"]:hover { border-color: var(--accent) !important; }

      /* Inputs */
      input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        background: #131722 !important;
        color: var(--fg) !important;
        font-family: 'Open Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 400 !important;
      }

      .stDataFrame { border: 1px solid var(--border); border-radius: 8px; }
      pre, code { background: var(--card) !important; color: var(--fg) !important; }

      /* room for sticky composer */
      .stApp > main { padding-bottom: 140px; }

      /* Chat bubbles */
      .chat-wrap { max-width: 880px; margin: 0 auto; }
      .msg { border: 1px solid var(--border); background: var(--card);
             border-radius: 16px; padding: 14px 16px; margin: 8px 0; line-height: 1.55; }
      .msg.assistant { background: rgba(105,163,218,0.08); }
      .msg.user { background: rgba(128,128,128,0.10); }

      /* Sticky composer */
      .composer { position: fixed; left: 50%; transform: translateX(-50%); bottom: 56px; width: min(880px, 90vw);
                  background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 10px; z-index: 50; }
      .composer .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }

      /* Sidebar sessions */
      .sidebar-inner { padding-top: 6px; }
      .sidebar-header { font-weight: 700; margin: 8px 0 6px; color: var(--muted); }
      .conv-btn { width: 100%; text-align: left; padding: 8px 10px; border: 1px solid var(--border);
                  border-radius: 8px; margin-bottom: 6px; background: var(--card); color: var(--fg); }
      .conv-btn:hover { border-color: var(--accent); }
    </style>
    """
    st.markdown(css_vars + base, unsafe_allow_html=True)


# =========================================================
# Query param helpers (tabs + thread)
# =========================================================
def _qp_get_one(key: str):
    v = st.query_params.get(key)
    if isinstance(v, list):
        return v[0] if v else None
    return v

def set_tab_param(val: str | None):
    if val is None:
        try: del st.query_params["tab"]
        except KeyError: pass
    else:
        st.query_params["tab"] = val

def get_tab_param() -> str | None:
    return _qp_get_one("tab")

def set_thread_param(val: str | None):
    if val is None:
        try: del st.query_params["thread"]
        except KeyError: pass
    else:
        st.query_params["thread"] = val

def get_thread_param() -> str | None:
    return _qp_get_one("thread")

# =========================================================
# Auth (SQLite + PBKDF2)
# =========================================================
AUTH_DB = "auth.db"
def _auth_conn():
    c = sqlite3.connect(AUTH_DB); c.row_factory = sqlite3.Row; return c
def init_auth_db():
    with _auth_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150_000).hex()
def create_user(username: str, password: str) -> bool:
    username = username.strip()
    if not username or not password: return False
    salt = secrets.token_hex(16); pwd_hash = _hash_password(password, salt)
    try:
        with _auth_conn() as con:
            con.execute("INSERT INTO users (username, hash, salt, created_at) VALUES (?, ?, ?, datetime('now'))",
                        (username, pwd_hash, salt))
        return True
    except sqlite3.IntegrityError:
        return False
def verify_user(username: str, password: str) -> bool:
    with _auth_conn() as con:
        cur = con.execute("SELECT hash, salt FROM users WHERE username = ?", (username.strip(),))
        row = cur.fetchone()
        if not row: return False
        calc = _hash_password(password, row["salt"])
        return secrets.compare_digest(calc, row["hash"])
def get_user(username: str):
    with _auth_conn() as con:
        cur = con.execute("SELECT id, username, created_at FROM users WHERE username = ?", (username.strip(),))
        return cur.fetchone()
def update_password(username: str, old_password: str, new_password: str) -> bool:
    with _auth_conn() as con:
        cur = con.execute("SELECT hash, salt FROM users WHERE username = ?", (username.strip(),))
        row = cur.fetchone()
        if not row: return False
        if not secrets.compare_digest(_hash_password(old_password, row["salt"]), row["hash"]):
            return False
        new_salt = secrets.token_hex(16)
        new_hash = _hash_password(new_password, new_salt)
        con.execute("UPDATE users SET hash=?, salt=? WHERE username=?", (new_hash, new_salt, username.strip()))
        return True
def delete_user(username: str, password: str) -> bool:
    if not verify_user(username, password): return False
    with _auth_conn() as con:
        con.execute("DELETE FROM users WHERE username=?", (username.strip(),))
    return True
init_auth_db()
def is_logged_in() -> bool: return bool(st.session_state.get("user"))
def current_user() -> str | None: return st.session_state.get("user")
def login(username: str): st.session_state["user"] = username
def logout(): st.session_state.pop("user", None)

# =========================================================
# App data (SQLite): Messenger + History
# =========================================================
APP_DB = "misfold_app.db"
def _app_conn():
    c = sqlite3.connect(APP_DB); c.row_factory = sqlite3.Row; return c
def init_app_db():
    with _app_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_key TEXT NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_key)")
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
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON history(username)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_accession ON history(accession)")
init_app_db()
def add_msg(thread_key: str, author: str, body: str):
    if not body or not body.strip(): return
    author = (author or "Analyst").strip()
    with _app_conn() as con:
        con.execute("INSERT INTO messages (thread_key, author, body, created_at) VALUES (?, ?, ?, datetime('now'))",
                    (thread_key, author, body.strip()))
def list_msgs(thread_key: str, limit: int = 200):
    with _app_conn() as con:
        cur = con.execute("SELECT author, body, created_at FROM messages WHERE thread_key = ? ORDER BY id DESC LIMIT ?",
                          (thread_key, limit))
        return list(cur.fetchall())
def add_history(username: str | None, filename: str, accession: str | None,
                error: float | None, threshold: float | None, misfold: int):
    with _app_conn() as con:
        con.execute("INSERT INTO history (username, filename, accession, error, threshold, misfold, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                    (username, filename, accession, error, threshold, misfold))
def list_history(username: str | None, limit: int = 200):
    with _app_conn() as con:
        if username:
            cur = con.execute("SELECT * FROM history WHERE username = ? ORDER BY id DESC LIMIT ?",
                              (username, limit))
            return list(cur.fetchall())
        else:
            return []  # hide history entirely when not logged in
def clear_history(username: str):
    with _app_conn() as con:
        con.execute("DELETE FROM history WHERE username = ?", (username,))

# =========================================================
# Model & inference (standalone AE)
# =========================================================
class FoldAutoencoder(nn.Module):
    def __init__(self, input_dim=300, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

@st.cache_resource
def load_model(model_path="trained_autoencoder.pt"):
    model = FoldAutoencoder()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval(); return model

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
                    scores.append(residue["CA"].bfactor)  # pLDDT in B-factor
    scores = scores[:max_len]
    if len(scores) < max_len:
        scores += [0.0] * (max_len - len(scores))
    return np.array(scores, dtype=np.float32)

def infer_misfold(model: FoldAutoencoder, raw_scores: np.ndarray, threshold: float = 0.03):
    x = (raw_scores / 100.0).astype(np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x_tensor)
        error = torch.mean((out - x_tensor) ** 2).item()  # per-residue MSE
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
            if len(lines) >= 20: break
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
            if i > 500: break
    return parse_uniprot_from_text(" ".join(text)) if text else None

@st.cache_data(show_spinner=False)
def fetch_uniprot_function(accession: str):
    if not accession: return None, None, None
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return None, None, f"https://www.uniprot.org/uniprotkb/{accession}"
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
# Build protein context (used by Chatbot too)
# =========================================================
def summarize_plddt(scores: np.ndarray) -> Dict[str, Any]:
    if scores is None or len(scores) == 0:
        return {"n": 0, "mean": None, "min": None, "max": None, "low_segments": []}
    s = np.array(scores, dtype=float); n = len(s)
    mean = float(np.mean(s)); minv = float(np.min(s)); maxv = float(np.max(s))
    low_segments: List[Tuple[int, int]] = []; in_seg = False; start = 0
    for i, v in enumerate(s):
        if v < 50 and not in_seg: in_seg = True; start = i
        if (v >= 50 or i == n - 1) and in_seg:
            end = i if v >= 50 else i
            low_segments.append((start + 1, end + 1)); in_seg = False
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

# =========================================================
# OpenAI backend only
# =========================================================
def call_openai_chat(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(api_key="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")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.2, max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

def answer_any_question(
    question: str,
    filename: str,
    accession: str | None,
    uni_name: str | None,
    uni_function: str | None,
    header_text: str | None,
    scores: np.ndarray | None,
    recon_error: float | None,
    threshold: float | None,
) -> str:
    protein_ctx = build_protein_context(
        filename, accession, uni_name, uni_function, header_text, scores, recon_error, threshold
    )
    system_prompt = (
        "You are a helpful assistant. Answer questions clearly and concisely.\n"
        "If the user's question is about protein structures, AlphaFold, pLDDT, or misfold detection, "
        "use the following context to ground your answer. If it's unrelated, ignore the context and "
        "answer normally.\n\n"
        f"Optional protein context:\n{protein_ctx}\n"
    )
    return call_openai_chat(system_prompt, question)

# =========================================================
# Header (login modal/popover only) + Footer (dynamic links)
# =========================================================
def login_signup_ui():
    tabs = st.tabs(["Login", "Sign up"])
    with tabs[0]:
        with st.form("login_form"):
            user = st.text_input("Username", key="login_user")
            pwd = st.text_input("Password", type="password", key="login_pwd")
            submit = st.form_submit_button("Login")
        if submit:
            if verify_user(user, pwd):
                login(user)
                st.success(f"Welcome, {user}!")
                st.session_state.pop("login_open", None)
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tabs[1]:
        with st.form("signup_form"):
            user2 = st.text_input("Choose a username", key="su_user")
            pwd2 = st.text_input("Choose a password", type="password", key="su_pwd")
            pwd3 = st.text_input("Confirm password", type="password", key="su_pwd2")
            submit2 = st.form_submit_button("Create account")
        if submit2:
            if not user2 or not pwd2 or pwd2 != pwd3:
                st.error("Please fill all fields and ensure passwords match.")
            else:
                ok = create_user(user2, pwd2)
                if ok:
                    st.success("Account created. You can now sign in.")
                else:
                    st.error("Username already exists.")

def maybe_open_login_dialog():
    if st.session_state.get("login_open") and hasattr(st, "dialog"):
        @st.dialog("Sign in")
        def _dlg():
            login_signup_ui()
        _dlg()

def render_header():
    # Title moved to sidebar; header only shows auth state + button
    st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)
    cols = st.columns([7,3,2])
    with cols[1]:
        if is_logged_in():
            st.markdown(f"**Signed in as:** `{current_user()}`")
    with cols[2]:
        if is_logged_in():
            if st.button("Logout", use_container_width=True):
                logout(); st.success("Logged out."); st.rerun()
        else:
            if hasattr(st, "dialog"):
                if st.button("Login", use_container_width=True):
                    st.session_state["login_open"] = True
            else:
                with st.popover("Login", use_container_width=True):
                    login_signup_ui()

def render_footer():
    # Build footer links dynamically, hiding History when logged out
    active = st.session_state.get("nav", get_tab_param() or "Results")
    tabs = ["Results", "Messenger", "Ask", "Account"] if not is_logged_in() else ["Results", "Messenger", "History", "Ask", "Account"]
    if active not in tabs:
        active = "Results"
        st.session_state["nav"] = active
        set_tab_param(active)

    def _link(tab):
        cls = "active" if tab == active else ""
        return f'<a class="{cls}" href="?tab={tab}">{tab}</a>'

    st.markdown(
        f'<div class="footer-bar">{" • ".join(_link(t) for t in tabs)}</div>',
        unsafe_allow_html=True
    )

# =========================================================
# Messenger UI
# =========================================================
def render_messenger(thread_key: str, default_body: str = ""):
    st.subheader("💬 Messenger (this protein)")
    st.caption(f"Thread key: `{thread_key}` — messages persist in a local SQLite DB.")
    if not is_logged_in():
        st.info("Please **sign in** to post messages.")
        st.text_area("Message (login required to send)", value=default_body, height=100, disabled=True)
        st.button("->", disabled=True)
    else:
        author = current_user() or "Analyst"
        body = st.text_area("Message", value=default_body, height=100, key=f"compose_{thread_key}")
        c1, c2, _ = st.columns([1,1,6])
        with c1:
            if st.button("->", key=f"send_{thread_key}"):
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
# Sidebar (title, tabs, sessions, controls)
# =========================================================
def render_sidebar():
    with st.sidebar:
        # --- Title moved to sidebar ---
        st.markdown("## Genome")
        st.caption("pLDDT • UniProt • Autoencoder misfold detection")

        # Sidebar tabs (hide History when not logged in)
        all_tabs = ["Results", "Messenger", "History", "Ask", "Account"]
        tabs = all_tabs if is_logged_in() else ["Results", "Messenger", "Ask", "Account"]

        current_default = st.session_state.get("nav", get_tab_param() or "Results")
        if current_default not in tabs:
            current_default = "Results"
        nav_choice = st.radio(
            "Navigate",
            options=tabs,
            index=tabs.index(current_default),
            label_visibility="collapsed",
        )
        st.session_state["nav"] = nav_choice
        set_tab_param(nav_choice)

        # Sessions list (hidden when not logged in)
        st.header("🗂 Sessions")
        if is_logged_in():
            if st.button("＋ New Analysis", use_container_width=True):
                for k in list(st.session_state.keys()):
                    if k.startswith("chat_") or k.startswith("_ae_model_"):
                        st.session_state.pop(k)
                st.query_params.clear()
                st.rerun()
            rows = list_history(current_user(), limit=12)
            st.markdown('<div class="sidebar-inner">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-header">Recent</div>', unsafe_allow_html=True)
            if rows:
                for r in rows:
                    label = (r["accession"] or r["filename"] or "session")
                    label = re.sub(r"\.pdb(\.gz)?$", "", label)
                    if st.button(label, key=f"h_{r['id']}", use_container_width=True, help=str(r["created_at"])):
                        set_thread_param(r["accession"] or r["filename"])
                        set_tab_param("Messenger")
                        st.rerun()
            else:
                st.caption("No sessions yet.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("Sign in to view your analysis history.")

        # Controls
        st.header("⚙️ Controls")
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
        with st.expander("About"):
            st.write(
                "Upload an AlphaFold structure (.pdb or .pdb.gz) to visualize per-residue pLDDT, "
                "fetch a UniProt function summary, and estimate potential misfolding via an autoencoder trained on pLDDT profiles."
            )
            st.write("• **pLDDT** is AlphaFold’s per-residue confidence score (0–100). Low pLDDT often indicates flexible/disordered regions.")
            st.markdown("### 🔬 What are PDB files?")
            st.write(
                "A **PDB file** stores 3D coordinates + metadata. AlphaFold PDBs store **pLDDT** in the B-factor column. "
                "This app extracts pLDDT, fetches UniProt function, and runs misfold detection."
            )
        return nav_choice, uploaded, model_path, model_exists, threshold, show_uniprot, show_raw_header

# =========================================================
# History & Account tabs
# =========================================================
def render_history_tab():
    st.subheader("🗂 History")
    user = current_user()
    if not user:
        st.info("Please sign in to view your history.")
        return
    rows = list_history(user, limit=500)
    st.caption("Showing **your** recent analyses.")
    if not rows:
        st.info("No history yet."); return
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

def render_account_tab():
    st.subheader("👤 Account")
    if not is_logged_in():
        st.info("You’re not signed in. Click **Login** in the top bar.")
        return
    u = get_user(current_user())
    st.markdown(f"**Username:** `{u['username']}`  •  **Created:** `{u['created_at']}`")
    st.markdown("---")
    st.markdown("### 🔑 Change Password")
    with st.form("change_pwd"):
        old_pwd = st.text_input("Current password", type="password")
        new_pwd = st.text_input("New password", type="password")
        new_pwd2 = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update password")
    if 'submitted' in locals() and submitted:
        if not new_pwd or new_pwd != new_pwd2:
            st.error("New passwords don’t match.")
        else:
            ok = update_password(u["username"], old_pwd, new_pwd)
            st.success("Password updated.") if ok else st.error("Current password is incorrect.")
    st.markdown("---")
    st.markdown("### ❌ Delete Account")
    with st.form("delete_acct"):
        confirm_user = st.text_input("Type your username to confirm")
        confirm_pwd = st.text_input("Password", type="password")
        agree = st.checkbox("I understand this is permanent.")
        dsub = st.form_submit_button("Delete my account")
    if 'dsub' in locals() and dsub:
        if confirm_user != u["username"] or not agree:
            st.error("Confirmation failed. Type your username and check the box.")
        else:
            if delete_user(u["username"], confirm_pwd):
                logout(); st.success("Account deleted."); st.rerun()
            else:
                st.error("Password incorrect — account not deleted.")

# =========================================================
# Chatbot tab — GENERAL QA (OpenAI only)
# =========================================================
def render_chatbot_tab(filename, accession, uni_name, uni_function, header_text, scores, recon_error, threshold):
    st.subheader("Ask")
    chat_key = f"chat_{accession or filename}"
    st.session_state.setdefault(chat_key, [])

    # Transcript
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    if not st.session_state[chat_key]:
        st.markdown(
            '<div class="msg assistant">Ask me anything — general knowledge, coding, science, or specifics '
            'about AlphaFold, pLDDT, thresholds, and misfold detection. I’ll use protein context if relevant.</div>',
            unsafe_allow_html=True
        )
    else:
        for turn in st.session_state[chat_key]:
            css_role = "user" if turn["role"] == "user" else "assistant"
            st.markdown(f'<div class="msg {css_role}">{turn["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Sticky composer
    st.markdown('<div class="composer">', unsafe_allow_html=True)
    cols = st.columns([1,8,1])
    with cols[0]: st.write("")
    with cols[1]:
        prompt = st.text_input(
            "Message",
            key=f"composer_{chat_key}",
            label_visibility="collapsed",
            placeholder="Ask anything (e.g., “Explain Transformers”, “Best way to pick a threshold?”, “What is CRISPR?”)",
            help="If your question is about proteins/misfolds, I’ll automatically use the current protein context."
        )
    with cols[2]:
        ask = st.button("->", key=f"ask_{chat_key}")
    st.markdown('<div class="hint">Tip: You can ask general questions — not just misfolds.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle send
    if ask and prompt.strip():
        user_q = prompt.strip()
        st.session_state[chat_key].append({"role": "user", "content": user_q})
        try:
            ans = answer_any_question(
                user_q, filename, accession, uni_name, uni_function, header_text, scores, recon_error, threshold
            )
            st.session_state[chat_key].append({"role": "assistant", "content": ans})
        except Exception as e:
            st.session_state[chat_key].append({
                "role": "assistant",
                "content": (
                    "General QA requires an OpenAI key. Please set environment variable `OPENAI_API_KEY`.\n\n"
                    f"Details: {e}"
                )
            })
        st.rerun()

# =========================================================
# Results page (OpenAI-like cards)
# =========================================================
def render_results_like_openai(uploaded, model_exists, model_path, threshold, show_uniprot, show_raw_header):
    st.markdown("""
    <style>
      .hero { padding: 72px 0 36px; text-align: center;
              background: radial-gradient(1200px 400px at 50% -200px, rgba(66,127,216,0.10), transparent 60%); }
      .hero h1 { font-size: clamp(32px, 5vw, 56px); line-height: 1.05; letter-spacing: -0.02em;
                 margin: 0 auto 14px; max-width: 900px; font-weight: 700; }
      .hero .sub { color: var(--muted); font-size: clamp(14px, 2vw, 18px);
                   margin: 0 auto 22px; max-width: 780px; }
      .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; margin: 24px auto; max-width: 1100px; }
      .card { grid-column: span 6; border:1px solid var(--border); border-radius:16px; background: var(--card);
              padding: 18px; min-height: 120px; }
      .card.wide { grid-column: 1 / -1; }
      .card h3 { margin: 0 0 8px; font-size: 18px; }
      .muted { color: var(--muted); }
      .metric { display:flex; gap: 16px; align-items:center; padding:12px; border:1px dashed var(--border);
                border-radius:12px; background: rgba(255,255,255,0.03); }
      .metric b { font-size: 20px; }
      .chip { display:inline-block; padding:4px 8px; border-radius:999px; border:1px solid var(--border); font-size:12px; }
      .pill { display:inline-block; padding:6px 10px; border:1px solid var(--border); border-radius:999px; font-size:12px; color: var(--muted); margin: 0 6px; }
      .kpi { display:flex; gap:14px; flex-wrap:wrap; }
      .kpi .pill { margin-top: 8px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub">Upload an AlphaFold PDB (or .pdb.gz). '
        'We’ll extract <b>pLDDT</b>, fetch <b>UniProt</b> function, and run a lightweight '
        '<b>autoencoder</b> to flag potential misfolds—presented in a clean, minimal UI.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        up2 = st.file_uploader("Drop a .pdb or .pdb.gz here", type=["pdb", "gz"], key="center_uploader", label_visibility="collapsed")
        if up2 and not uploaded:
            uploaded = up2
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="grid">', unsafe_allow_html=True)

    if uploaded is None:
        with st.container():
            st.markdown(
                """
                <div class="card"><h3>1. Upload</h3>
                <div class="muted">Provide a PDB or .pdb.gz. AlphaFold stores pLDDT in the B-factor column (CA atoms).</div></div>
                """, unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="card"><h3>2. Analyze</h3>
                <div class="muted">We extract pLDDT, summarize confidence, and (optionally) fetch UniProt function text.</div></div>
                """, unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="card"><h3>3. Detect</h3>
                <div class="muted">A small autoencoder computes reconstruction error. If error > threshold → flagged.</div></div>
                """, unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="card"><h3>4. Share</h3>
                <div class="muted">Discuss results in <b>Messenger</b>, save to <b>History</b>, or ask questions in <b>Ask</b>.</div></div>
                """, unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    tmp_path = f"temp_{uploaded.name}"
    with open(tmp_path, "wb") as f: f.write(uploaded.read())

    accession = None; description = None; uni_name = None; function_text = None
    scores = None; error = None; is_mf_flag = None

    try:
        description = get_protein_description(tmp_path)
        accession = extract_uniprot_accession(tmp_path)
    except Exception as e:
        st.warning(f"Could not parse header/accession: {e}")

    with st.container():
        st.markdown('<div class="card wide">', unsafe_allow_html=True)
        st.markdown("#### Protein summary")
        left, right = st.columns([2,3])
        with left:
            if description and description.strip():
                if show_raw_header:
                    st.code(description, language="text")
                else:
                    lines = description.splitlines()
                    st.write("\n".join(lines[:6] if len(lines) > 6 else lines))
                    if len(lines) > 6: st.caption("Tip: enable **Show raw PDB header lines** in the sidebar.")
            else:
                st.info("No PDB header description found.")
        with right:
            st.markdown("**Identifiers**")
            st.write(f"- **File**: `{uploaded.name}`")
            st.write(f"- **Accession**: `{accession or 'unknown'}`")
            st.write(f"- **Model path**: `{model_path}`")
            st.write(f"- **Threshold**: `{threshold:.3f}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### UniProt function")
        if show_uniprot:
            if accession:
                function_text, uni_name, uni_url = fetch_uniprot_function(accession)
                header = f"**{uni_name or accession}**  •  Accession: `{accession}`"
                if uni_url: header += f"  •  [View on UniProt]({uni_url})"
                st.markdown(header)
                st.write(function_text) if function_text else st.info("No function text available from UniProt for this accession.")
            else:
                st.warning("Couldn’t determine a UniProt accession from the filename/header.")
        else:
            st.caption("UniProt lookup disabled.")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### pLDDT profile")
        try:
            scores = extract_plddt_scores(tmp_path)
            st.line_chart(scores, use_container_width=True)
            s = summarize_plddt(scores)
            st.markdown(
                f'<div class="kpi">'
                f'<span class="pill">Residues: <b>{s["n"]}</b></span>'
                f'<span class="pill">Mean: <b>{s["mean"]:.2f}</b></span>'
                f'<span class="pill">Min: <b>{s["min"]:.2f}</b></span>'
                f'<span class="pill">Max: <b>{s["max"]:.2f}</b></span>'
                f'</div>', unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Failed to extract pLDDT scores: {e}"); scores = None
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card wide">', unsafe_allow_html=True)
        st.markdown("#### Misfold detection")
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
                st.markdown(
                    f'<div class="metric"><b>{error:.4f}</b> '
                    f'<span class="muted">reconstruction error</span>'
                    f' <span class="chip">threshold {threshold:.3f}</span></div>',
                    unsafe_allow_html=True
                )
                st.error("⚠️ Likely Misfolded (error > threshold)") if is_misfold else st.success("✅ Structure appears typical (error ≤ threshold)")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    try:
        if scores is not None and error is not None and is_mf_flag is not None:
            add_history(current_user(), uploaded.name, accession, float(error), float(threshold), int(is_mf_flag))
    except Exception as e:
        st.warning(f"Could not record history: {e}")

    try: os.remove(tmp_path)
    except Exception: pass

# =========================================================
# Main App (sidebar tabs)
# =========================================================
def main():
    apply_theme("dark")
    render_header()

    # Sync selected tab from URL if present
    tab_q = get_tab_param()
    allowed_tabs = ["Results", "Messenger", "History", "Ask", "Account"] if is_logged_in() else ["Results", "Messenger", "Ask", "Account"]
    if tab_q in allowed_tabs:
        st.session_state["nav"] = tab_q
    elif tab_q and tab_q not in allowed_tabs:
        st.session_state["nav"] = "Results"
        set_tab_param("Results")

    nav, uploaded, model_path, model_exists, threshold, show_uniprot, show_raw_header = render_sidebar()

    if nav == "Results":
        render_results_like_openai(
            uploaded=uploaded,
            model_exists=model_exists,
            model_path=model_path,
            threshold=threshold,
            show_uniprot=show_uniprot,
            show_raw_header=show_raw_header,
        )

    elif nav == "Messenger":
        thread_key = get_thread_param()
        if not thread_key:
            rows = list_history(current_user(), limit=1) if is_logged_in() else []
            if rows:
                r0 = rows[0]; thread_key = r0["accession"] or r0["filename"]
        render_messenger(thread_key or "general", default_body="")

    elif nav == "History":
        render_history_tab()

    elif nav == "Ask":
        filename = "no_file"; acc = None; u_name = None; u_func = None; header = None; sc = None; err = None
        thr = threshold if "threshold" in locals() else None
        render_chatbot_tab(filename, acc, u_name, u_func, header, sc, err, thr)

    elif nav == "Account":
        render_account_tab()

    render_footer()
    maybe_open_login_dialog()

if __name__ == "__main__":
    main()
