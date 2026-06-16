import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
from PIL import Image

PREDICTIONS_DIR = Path("data/predictions")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_data_contract(
    image_bytes: bytes,
    image_filename: str,
    preds: list,
    model_name: str,
    confidence_threshold: float = 0.9,
) -> dict:
    """Build an HTR data contract JSON from Kraken predictions."""
    doc_id = Path(image_filename).stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = []
    for i, p in enumerate(preds, start=1):
        text = p.prediction or ""
        char_confs = list(p.confidences) if p.confidences else []
        confidence = sum(char_confs) / len(char_confs) if char_confs else 0.0

        import statistics
        char_std = statistics.pstdev(char_confs) if len(char_confs) > 1 else 0.0
        needs_review = confidence < confidence_threshold or char_std > 0.2

        polygon = []
        if hasattr(p, "cuts") and p.cuts:
            try:
                polygon = [[int(x), int(y)] for x, y in p.cuts]
            except Exception:
                polygon = []

        lines.append({
            "line_id": f"{doc_id}_l{i:03d}",
            "text": text,
            "confidence": round(confidence, 4),
            "char_confidences": [round(c, 4) for c in char_confs],
            "candidates": None,
            "needs_review": needs_review,
            "polygon": polygon,
            "reading_order": i,
        })

    return {
        "document_id": doc_id,
        "metadata": {
            "source": image_filename,
            "century_estimate": "XIII",
            "document_type": "unknown",
            "scan_quality": "unknown",
            "sha256": _sha256_bytes(image_bytes),
            "model": model_name,
            "produced_at": datetime.now().isoformat(timespec="seconds"),
        },
        "pages": [
            {
                "page_id": "p001",
                "image_path": image_filename,
                "lines": lines,
            }
        ],
    }


def save_data_contract(contract: dict) -> Path:
    """Save data contract JSON to data/predictions/ and return the path."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{contract['document_id']}.json"
    out_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path

HF_REPO = "legb/htr-cremma-medieval"

MODELS = {
    "Exp 2 — Baseline binarisée (CER 26.3%)": {
        "filename": "exp2_binarise_20260613.safetensors",
        "cer": 0.2633,
        "val_accuracy": 0.7367,
        "stages": 37,
        "plateforme": "Kaggle T4 x2",
        "date": "13 juin 2026",
        "donnees": "train.arrow — binarisé mode 1 (939 MB)",
        "modele_base": "cremma_generic.mlmodel",
        "batch": 8,
        "precision": "16-mixed (fp16)",
        "lag": 10,
        "note": "Plafond ~74% dû au mismatch mode L/1 — données binarisées vs modèle entraîné en grayscale.",
        "commit_hf": "99843b75",
    },
    "Exp 3 — Arrow filtré grayscale (en cours)": {
        "filename": "exp3_clean_arrow_20260613.safetensors",
        "cer": None,
        "val_accuracy": None,
        "stages": None,
        "plateforme": "Kaggle T4 x2",
        "date": "en cours",
        "donnees": "train_clean.arrow — grayscale mode L filtré (914 MB, 18 769 lignes)",
        "modele_base": "cremma-generic-1.0.1.mlmodel",
        "batch": 8,
        "precision": "16-mixed (fp16)",
        "lag": 10,
        "note": "Zones bruit exclues (MusicZone, DropCapital, Interlinear). Premier test grayscale réel.",
        "commit_hf": "5e43b1b1",
    },
}


@st.cache_resource(show_spinner="Téléchargement et chargement du modèle HTR...")
def get_htr_model(filename: str):
    from huggingface_hub import hf_hub_download
    from kraken.models import load_safetensors
    from kraken.lib.models import TorchSeqRecognizer

    local_path = hf_hub_download(repo_id=HF_REPO, filename=filename)
    vgsl_models = load_safetensors(local_path, tasks=["recognition"])
    net = TorchSeqRecognizer(vgsl_models[0], device="cpu")
    return net


st.set_page_config(page_title="HTR CREMMA Medieval", layout="centered")
st.title("HTR Manuscrit XIIIe siècle")
st.caption("Fine-tuning Kraken — Ouazar, Tessier, El Mortada")

# --- Sidebar ---
st.sidebar.header("Modèle HTR")
model_name = st.sidebar.selectbox("Choisir un modèle", list(MODELS.keys()))
meta = MODELS[model_name]

st.sidebar.markdown("---")
st.sidebar.markdown("#### Métriques du modèle")

if meta["cer"] is not None:
    st.sidebar.metric("CER (val)", f"{meta['cer']:.1%}")
    st.sidebar.metric("val_accuracy", f"{meta['val_accuracy']:.1%}")
    st.sidebar.metric("Stages", meta["stages"])
else:
    st.sidebar.info("Résultats en attente (run en cours)")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Configuration d'entraînement")
st.sidebar.markdown(f"""
| Paramètre | Valeur |
|-----------|--------|
| Date | {meta['date']} |
| Plateforme | {meta['plateforme']} |
| Modèle base | `{meta['modele_base']}` |
| Données | {meta['donnees']} |
| Batch | {meta['batch']} |
| Precision | {meta['precision']} |
| Early stop lag | {meta['lag']} |
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"[HuggingFace](https://huggingface.co/{HF_REPO}) · "
    f"[GitHub](https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle)"
)

# --- Main ---
st.header("Page manuscrite à transcrire")

if meta["note"]:
    st.caption(f"Note : {meta['note']}")

uploaded = st.file_uploader(
    "Déposer une image de page ou de ligne (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="Image chargée (mode L)", width="stretch")

    if st.button("Segmenter et transcrire", type="primary"):
        try:
            from kraken import blla, rpred

            # 1. Segmentation
            with st.spinner("Segmentation des lignes (BLLA)..."):
                segmentation = blla.segment(image, device="cpu")

            n_lines = len(segmentation.lines) if segmentation.lines else 0
            st.info(f"{n_lines} ligne(s) détectée(s) par le segmenteur BLLA")

            if n_lines == 0:
                st.warning("Aucune ligne détectée. Vérifiez la qualité de l'image.")
            else:
                # 2. Modèle HTR
                net = get_htr_model(meta["filename"])

                # 3. Transcription
                with st.spinner(f"Transcription de {n_lines} ligne(s)..."):
                    preds = list(rpred.rpred(net, image, segmentation))

                lines_text = [p.prediction for p in preds if p.prediction.strip()]
                all_conf = [c for p in preds for c in (p.confidences or [])]
                avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0

                if lines_text:
                    st.success(f"{len(lines_text)} ligne(s) transcrite(s)")

                    # Export automatique data contract
                    image_bytes = uploaded.getvalue()
                    contract = build_data_contract(
                        image_bytes=image_bytes,
                        image_filename=uploaded.name,
                        preds=preds,
                        model_name=meta["filename"],
                    )
                    contract_path = save_data_contract(contract)
                    st.info(f"Data contract sauvegardé : `{contract_path}`")

                    # Métriques de l'inférence
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Lignes transcrites", len(lines_text))
                    col2.metric("Confiance moy.", f"{avg_conf:.1%}")
                    col3.metric("CER modèle (val)", f"{meta['cer']:.1%}" if meta['cer'] else "—")

                    st.markdown("### Transcription")
                    st.text("\n".join(lines_text))

                    # Détail par ligne
                    with st.expander("Détail ligne par ligne avec confiance"):
                        for i, p in enumerate(preds):
                            if p.prediction.strip():
                                conf_line = (
                                    sum(p.confidences) / len(p.confidences)
                                    if p.confidences else 0.0
                                )
                                st.markdown(
                                    f"**Ligne {i+1}** *(conf. {conf_line:.1%})* — `{p.prediction}`"
                                )

                    st.download_button(
                        label="Télécharger la transcription (.txt)",
                        data="\n".join(lines_text),
                        file_name="transcription.txt",
                        mime="text/plain",
                    )
                    st.download_button(
                        label="Télécharger le data contract (.json)",
                        data=json.dumps(contract, ensure_ascii=False, indent=2),
                        file_name=f"{contract['document_id']}.json",
                        mime="application/json",
                    )
                else:
                    st.warning("Lignes détectées mais aucun texte transcrit.")

        except Exception as e:
            st.error(f"Erreur : {e}")
            with st.expander("Traceback complet"):
                import traceback
                st.code(traceback.format_exc())

# --- Footer ---
st.markdown("---")
col1, col2 = st.columns(2)
col1.markdown("""
**Corpus** : Manuscrits XIIIe siècle (HTR-United)
**Période** : XIIIe siècle
**Langues** : Ancien français + Latin
**Objectif CER** : < 15% (validation) / < 8% (excellence)
""")
col2.markdown(f"""
**Modèle actif** : `{meta['filename']}`
**Commit HF** : `{meta['commit_hf']}`
**Pipeline** : BLLA segmentation → Kraken OCR
**Licence** : CC-BY 4.0
""")
