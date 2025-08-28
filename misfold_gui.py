import os
import re
import gzip
import json
import time
import hashlib
import secrets
import sqlite3
import requests
import pandas as pd
import streamlit as st
import torch
from datetime import datetime
from Bio.PDB import PDBParser

from misfold_inference import (
    FoldAutoencoder,
    extract_plddt_scores,
    infer_misfold,
)

# =========================
# Database (SQLite) helpers
# =========================

DB_PATH = os.environ.get("MISFOLD_DB_PATH", "misfold_app.db")

def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            uniprot_acc TEXT,
            n_residues INTEGER,
            plddt_mean REAL,
            plddt_min REAL,
            plddt_max REAL,
            recon_error REAL,
            threshold REAL,
            is_misfold INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)
        conn.commit()

def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def create_user(username: str, password: str):
    username = username.strip().lower()
    salt = secrets.token_hex(16)
    pw_hash = hash_password(password, salt)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, pw_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                    (username, pw_hash, salt, datetime.utcnow().isoformat()))
        conn.commit()

def get_user(username: str):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, pw_hash, salt FROM users WHERE username = ?", (username.strip().lower(),))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "username": row[1], "pw_hash": row[2], "salt": row[3]}
        return None

def authenticate(username: str, password: str):
    u = get_user(username)
    if not u: return None
    if hash_password(password, u["salt"]) == u["pw_hash"]:
        return u
    return None

def record_upload(user_id: int, filename: str, uniprot_acc: str,
                  plddt_scores, recon_error: float, threshold: float, is_misfold: bool):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO uploads (user_id, filename, uniprot_acc, n_residues,
                                 plddt_mean, plddt_min, plddt_max, recon_error,
                                 threshold, is_misfold, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, filename, uniprot_acc,
            int(len(plddt_scores)),
            float(pd.Series(plddt_scores).mean()),
            float(pd.Series(plddt_scores).min()),
            float(pd.Series(plddt_scores).max()),
            float(recon_error),
            float(threshold),
            int(bool(is_misfold)),
            datetime.utcnow().isoformat()
        ))
        conn.commit()

def fetch_history(user_id: int) -> pd.DataFrame:
    with db_conn() as conn:
        df = pd.read_sql_query("""
            SELECT filename, uniprot_acc, n_residues,
                   ROUND(plddt_mean,2) AS plddt_mean,
                   ROUND(plddt_min,2)  AS plddt_min,
                   ROUND(plddt_max,2)  AS plddt_max,
                   ROUND(recon_error,4) AS recon_error,
                   ROUND(threshold,4)    AS threshold,
                   CASE is_misfold WHEN 1 THEN 'Yes' ELSE 'No' END AS likely_misfold,
                   created_at
            FROM uploads
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, conn, params=(user_id,))
    return df

def clear_history(user_id: int):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM uploads WHERE user_id = ?", (user_id,))
        conn.commit()

# =========================
# AlphaFold helpers (same)
# =========================

def get_protein_description(file_path):
    opener = gzip.open if file_path.endswith(".gz") else open
    description = []
    with opener(file_path, "rt") as handle:
        for line in handle:
            if line.startswith(("HEADER", "TITLE", "COMPND")):
                description.append(line.strip())
            if len(description) > 20:
                break
    return "\n".join(description) if description else "No description found in PDB file."

def parse_uniprot_from_text(text: str):
    m = re.search(r'\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b', text)
    if m: return m.group(0)
    m10 = re.search(r'\b[A-NR-Z0-9]{1}[0-9][A-Z0-9]{3}[0-9][A-Z0-9]{4}\b', text)
    return m10.group(0) if m10 else None

def extract_uniprot_accession(file_path):
    fname = os.path.basename(file_path)
    m = re.search(r'AF-([A-Z0-9]+)-F\d+', fname)
    if m: return m.group(1)
    acc = parse_uniprot_from_text(fname)
    if acc: return acc
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt") as handle:
        header_text = []
        for line in handle:
            if line.startswith(("HEADER", "TITLE", "COMPND", "DBREF", "REMARK")):
                header_text.append(line)
            if len(header_text) > 200:
                break
    return parse_uniprot_from_text(" ".join(header_text)) if header_text else None

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
        # name
        try:
            name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
        except Exception:
            name = data.get("primaryAccession", accession)
        # function
        function_text = None
        for c in data.get("comments", []):
            if c.get("commentType") == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    function_text = texts[0].get("value")
                    break
        if function_text is None:
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

@st.cache_resource
def load_model(model_path="trained_autoencoder.pt"):
    model = FoldAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="AlphaFold Misfold Prediction Tool", layout="wide")
init_db()

# --- Auth state ---
if "user" not in st.session_state:
    st.session_state.user = None

with st.sidebar:
    st.header("👤 Account")
    if st.session_state.user is None:
        tabs = st.tabs(["Login", "Sign up"])
        with tabs[0]:
            login_user = st.text_input("Username", key="login_username")
            login_pw   = st.text_input("Password", type="password", key="login_pw")
            if st.button("Log in"):
                u = authenticate(login_user, login_pw)
                if u:
                    st.session_state.user = {"id": u["id"], "username": u["username"]}
                    st.success(f"Logged in as @{u['username']}")
                    st.rerun()

                else:
                    st.error("Invalid username or password.")
        with tabs[1]:
            su_user = st.text_input("New username", key="su_username")
            su_pw   = st.text_input("New password", type="password", key="su_pw")
            su_pw2  = st.text_input("Confirm password", type="password", key="su_pw2")
            if st.button("Create account"):
                if not su_user or not su_pw:
                    st.warning("Please provide username & password.")
                elif su_pw != su_pw2:
                    st.error("Passwords do not match.")
                elif get_user(su_user):
                    st.error("Username already exists.")
                else:
                    try:
                        create_user(su_user, su_pw)
                        st.success("Account created. You can log in now.")
                    except sqlite3.IntegrityError:
                        st.error("Username already exists.")
    else:
        st.write(f"Logged in as **@{st.session_state.user['username']}**")
        if st.button("Log out"):
            st.session_state.user = None
            st.rerun()

    st.markdown("---")
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload .pdb or .pdb.gz", type=["pdb", "gz"])
    model_path = st.text_input("Model path", value="trained_autoencoder.pt")
    model_ok = os.path.exists(model_path)
    st.caption(("✅ Found model" if model_ok else "❌ Model not found") + f": `{model_path}`")
    threshold = st.slider("Misfold threshold (reconstruction MSE)",
                          min_value=0.001, max_value=0.2, value=0.01, step=0.001)
    show_uniprot = st.checkbox("Fetch UniProt function", value=True)
    show_raw_header = st.checkbox("Show raw PDB header lines", value=False)

st.title("🧬 AlphaFold Misfold Prediction Tool")

# --- History (only for logged-in users) ---
if st.session_state.user is not None:
    st.subheader("📚 Your Upload History")
    hist_df = fetch_history(st.session_state.user["id"])
    if hist_df.empty:
        st.caption("No uploads recorded yet.")
    else:
        st.dataframe(hist_df, use_container_width=True)
        st.download_button(
            "Download history as CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name=f"uploads_{st.session_state.user['username']}.csv",
            mime="text/csv"
        )
        if st.button("Clear my history"):
            clear_history(st.session_state.user["id"])
            st.success("History cleared.")
            st.rerun()
else:
    st.info("Log in (left sidebar) to save and view your upload history.")

# --- Main workflow ---
if uploaded_file is None:
    st.info("Upload a `.pdb` or `.pdb.gz` file in the sidebar to begin.")
else:
    tmp_path = f"temp_{uploaded_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        st.subheader("📄 Protein Description")
        description = get_protein_description(tmp_path)
        if description.strip():
            if show_raw_header:
                st.code(description, language="text")
            else:
                lines = description.splitlines()
                st.write("\n".join(lines[:5]))
                if len(lines) > 5:
                    st.caption("…toggle “Show raw PDB header lines” to view all header lines.")

        accession = None
        if show_uniprot:
            accession = extract_uniprot_accession(tmp_path)
            st.subheader("🧾 UniProt Function")
            if accession:
                function_text, uni_name, uni_url = fetch_uniprot_function(accession)
                header = f"**{uni_name or accession}**  •  Accession: `{accession}`"
                if uni_url:
                    header += f"  •  [View on UniProt]({uni_url})"
                st.markdown(header)
                if function_text:
                    st.write(function_text)
                else:
                    st.info("No function text available from UniProt for this accession.")
            else:
                st.warning("Couldn’t determine a UniProt accession from the filename/header.")

        st.subheader("📊 pLDDT Score Profile")
        scores = extract_plddt_scores(tmp_path)
        st.line_chart(scores, use_container_width=True)

        if not model_ok:
            st.error(f"Model not found at `{model_path}`. Provide a valid path in the sidebar.")
        else:
            model = load_model(model_path)
            is_misfold, error = infer_misfold(model, scores, threshold=threshold)
            st.subheader("🔍 Prediction Result")
            st.markdown(f"**Reconstruction Error**: `{error:.4f}`  •  **Threshold**: `{threshold:.3f}`")
            if is_misfold:
                st.error("⚠️ Likely Misfolded (error > threshold)")
            else:
                st.success("✅ Structure appears typical (error ≤ threshold)")

            # Record upload if logged in
            if st.session_state.user is not None:
                try:
                    record_upload(
                        user_id=st.session_state.user["id"],
                        filename=uploaded_file.name,
                        uniprot_acc=accession,
                        plddt_scores=scores,
                        recon_error=error,
                        threshold=threshold,
                        is_misfold=is_misfold
                    )
                except Exception as e:
                    st.warning(f"Could not record upload: {e}")

    except Exception as e:
        st.error(f"Failed to process file: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
