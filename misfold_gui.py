import os
import re
import gzip
import json
import requests
import streamlit as st
import torch
from Bio.PDB import PDBParser
from misfold_inference import FoldAutoencoder, extract_plddt_scores, infer_misfold

# ---------- Helpers: description & accession parsing ----------

def get_protein_description(file_path):
    """Read HEADER/TITLE/COMPND lines from PDB or PDB.GZ."""
    opener = gzip.open if file_path.endswith(".gz") else open
    description = []
    with opener(file_path, "rt") as handle:
        for line in handle:
            if line.startswith(("HEADER", "TITLE", "COMPND")):
                description.append(line.strip())
            # stop after a bit; PDB headers can be long
            if len(description) > 20:
                break
    return "\n".join(description) if description else "No description found in PDB file."

def parse_uniprot_from_text(text: str):
    """
    Try to find a UniProt accession in text (supports most common 6-char IDs).
    Patterns include: P69905, Q9Y2Z4, A0A123 (10-char new style isn't covered here—handled below too).
    """
    # Common 6-char accessions (two regex alternatives collapsed with |)
    m = re.search(r'\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b', text)
    if m:
        return m.group(0)

    # Also try 10-char accessions (e.g., A0A024RBG1)
    m10 = re.search(r'\b[A-NR-Z0-9]{1}[0-9][A-Z0-9]{3}[0-9][A-Z0-9]{4}\b', text)
    if m10:
        return m10.group(0)

    return None

def extract_uniprot_accession(file_path):
    """
    Heuristics:
      1) Parse filename like AF-P69905-F1*.pdb(.gz) → P69905
      2) Parse any UniProt-looking token in filename
      3) Scan the header lines for an accession
    """
    fname = os.path.basename(file_path)

    # AlphaFold DB filenames often look like: AF-P69905-F1-model_v4.pdb.gz
    m = re.search(r'AF-([A-Z0-9]+)-F\d+', fname)
    if m:
        return m.group(1)

    # Try any UniProt-looking token from filename
    acc = parse_uniprot_from_text(fname)
    if acc:
        return acc

    # Fallback: scan header for accession
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt") as handle:
        header_text = []
        for line in handle:
            if line.startswith(("HEADER", "TITLE", "COMPND", "DBREF", "REMARK")):
                header_text.append(line)
            if len(header_text) > 200:
                break
    return parse_uniprot_from_text(" ".join(header_text)) if header_text else None

# ---------- UniProt function fetch ----------

@st.cache_data(show_spinner=False)
def fetch_uniprot_function(accession: str):
    """
    Query UniProt REST API for function text.
    Returns (function_text, uniprot_name, uniprot_url) or (None, None, None) on failure.
    """
    if not accession:
        return None, None, None
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None, None, f"https://www.uniprot.org/uniprotkb/{accession}"
        data = r.json()

        # Recommended protein name (if present)
        name = None
        try:
            name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
        except Exception:
            # fallback to uniProtkb id
            name = data.get("primaryAccession", accession)

        # Function comment(s)
        function_text = None
        comments = data.get("comments", [])
        for c in comments:
            if c.get("commentType") == "FUNCTION":
                # A function entry may have multiple texts
                texts = c.get("texts", [])
                if texts:
                    function_text = texts[0].get("value")
                    break

        # As a backup, pull GO-MF terms if function text missing
        if function_text is None:
            go_terms = []
            for dbref in data.get("uniProtKBCrossReferences", []):
                if dbref.get("database") == "GO":
                    props = {p["key"]: p["value"] for p in dbref.get("properties", [])}
                    # mf = molecular function
                    if props.get("aspect") == "F":
                        go_terms.append(props.get("term"))
            if go_terms:
                function_text = "GO molecular function terms: " + "; ".join(go_terms)

        return function_text, name, f"https://www.uniprot.org/uniprotkb/{accession}"
    except Exception:
        return None, None, f"https://www.uniprot.org/uniprotkb/{accession}"

# ---------- Model loader ----------

@st.cache_resource
def load_model():
    model = FoldAutoencoder()
    model.load_state_dict(torch.load("trained_autoencoder.pt", map_location="cpu"))
    model.eval()
    return model

# ---------- UI ----------

st.title("Misfold Prediction Tool")
st.markdown("This app is an interactive web tool for exploring AlphaFold-predicted protein structures and evaluating their likelihood of misfolding. It combines structural confidence metrics (pLDDT) from AlphaFold with a machine learning model (autoencoder) to highlight stable vs. unstable regions of a protein.")
st.markdown("Upload an AlphaFold **.pdb** or **.pdb.gz** file to view its description, **UniProt function**, and a misfold prediction.")

uploaded_file = st.file_uploader("Upload File", type=["pdb", "gz"])

if uploaded_file is not None:
    # Save to a temp file
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded successfully!")

    # 1) Show description from header
    st.subheader("📄 Protein Description (from PDB header)")
    description = get_protein_description(file_path)
    st.text(description)

    # 2) UniProt function lookup
    accession = extract_uniprot_accession(file_path)
    st.subheader("🧾 UniProt Function")
    if accession:
        function_text, uni_name, uni_url = fetch_uniprot_function(accession)
        if uni_name:
            st.markdown(f"**{uni_name}**  •  Accession: `{accession}`  •  [View on UniProt]({uni_url})")
        else:
            st.markdown(f"Accession: `{accession}`  •  [UniProt page]({uni_url})")

        if function_text:
            st.write(function_text)
        else:
            st.info("No function text available from UniProt for this accession.")
    else:
        st.warning("Couldn’t determine a UniProt accession from the file. "
                   "Rename the file to include the accession (e.g., `AF-P69905-F1...`) "
                   "or ensure the header has accession info.")

    # 3) pLDDT plot + prediction
    try:
        scores = extract_plddt_scores(file_path)
        st.subheader("📊 pLDDT Score Profile")
        st.line_chart(scores, use_container_width=True)

        model = load_model()
        is_misfold, error = infer_misfold(model, scores)
        st.subheader("🔍 Prediction Result")
        st.markdown(f"**Reconstruction Error**: `{error:.4f}`")
        if is_misfold:
            st.error("⚠️ Likely Misfolded!")
        else:
            st.success("✅ Structure appears typical.")
    except Exception as e:
        st.error(f"Failed to process file: {e}")
    finally:
        os.remove(file_path)