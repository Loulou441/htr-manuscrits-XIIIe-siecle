import os
import re
import sys
import json
import requests
from pathlib import Path

# On s'assure que le répertoire courant est dans le path pour importer app.py
sys.path.append(os.getcwd())

try:
    import app
    from kraken import blla, rpred
    from PIL import Image
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    print("Assurez-vous d'avoir installé kraken, streamlit, pillow, requests et d'exécuter le script dans le dossier de app.py.")
    sys.exit(1)

# Configuration
TXT_FILE = "Manuscrits XIII siecle.txt"
OUTPUT_DIR = Path("data/predictions")
DOWNLOAD_DIR = Path("data/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def extract_gallica_ark(url):
    """Extrait l'identifiant ARK de Gallica à partir d'une URL."""
    match = re.search(r'(ark:/[0-9]+/btv[a-b0-9]+)', url)
    if match:
        return match.group(1)
    return None

def download_gallica_page(ark_id, page_num):
    """Télécharge une page spécifique de Gallica au format JPEG de haute qualité."""
    # Gallica utilise une numérotation à base 1 pour l'accès direct via l'API de calcul
    # Format de l'URL d'image standard Gallica IIIF / Raccourci natif:
    download_url = f"https://gallica.bnf.fr/{ark_id}/f{page_num}.highres"
    
    local_filename = DOWNLOAD_DIR / f"{ark_id.split('/')[-1]}_f{page_num}.jpg"
    
    if local_filename.exists():
        return local_filename
        
    try:
        response = requests.get(download_url, timeout=30)
        if response.status_code == 200 and b"html" not in response.content[:100]:
            local_filename.write_bytes(response.content)
            return local_filename
    except Exception as e:
        print(f"  ⚠️ Erreur lors du téléchargement de la page {page_num} : {e}")
    return None

def process_batch():
    # 1. Extraction des manuscrits du fichier texte
    if not Path(TXT_FILE).exists():
        print(f"❌ Le fichier {TXT_FILE} est introuvable.")
        return

    content = Path(TXT_FILE).read_text(encoding="utf-8")
    # Recherche de toutes les lignes contenant un lien Gallica btv
    lines = content.splitlines()
    manuscripts = []
    
    for line in lines:
        if "https://gallica.bnf.fr" in line:
            # Nettoyage rapide du nom
            name = line.split(":")[0].replace("", "").replace("", "").replace("", "").strip()
            url = line.split("https://")[1]
            url = "https://" + url
            ark_id = extract_gallica_ark(url)
            if ark_id:
                manuscripts.append({"name": name if name else "Manuscrit_Inconnu", "ark_id": ark_id})

    # Sélection des 10 premiers manuscrits uniques
    seen = set()
    unique_manuscripts = []
    for m in manuscripts:
        if m["ark_id"] not in seen:
            seen.add(m["ark_id"])
            unique_manuscripts.append(m)
        if len(unique_manuscripts) == 10:
            break

    print(f"📚 {len(unique_manuscripts)} manuscrits uniques trouvés. Lancement du traitement par lot...")

    # 2. Chargement du modèle par défaut de l'application (Exp 2)
    model_meta = app.MODELS["Exp 2 — Baseline binarisée (CER 26.3%)"]
    print(f"⏳ Téléchargement et chargement du modèle Kraken `{model_meta['filename']}`...")
    net = app.get_htr_model(model_meta["filename"])
    print("✅ Modèle chargé avec succès sur le CPU.")

    # 3. Boucle sur les manuscrits et leurs pages
    for idx, ms in enumerate(unique_manuscripts, start=1):
        print(f"\n📖 [{idx}/10] Traitement : {ms['name']} ({ms['ark_id']})")
        
        for page in range(1, 11): # Pages 1 à 10
            print(f"  📄 Page {page}/10...", end="", flush=True)
            
            img_path = download_gallica_page(ms["ark_id"], page)
            if not img_path:
                print(" Échec (Lien invalide ou fin du document).")
                continue
                
            try:
                # Ouverture et conversion en niveaux de gris (Mode L) requis par app.py
                image = Image.open(img_path).convert("L")
                
                # Étape 1 : Segmentation native de Kraken (BLLA)
                segmentation = blla.segment(image, device="cpu")
                
                if not segmentation.lines:
                    print(" Ignorée (Aucune ligne détectée).")
                    continue
                
                # Étape 2 : Transcription par reconnaissance
                preds = list(rpred.rpred(net, image, segmentation))
                
                # Étape 3 : Construction du Data Contract (Fonction de app.py)
                image_bytes = img_path.read_bytes()
                contract = app.build_data_contract(
                    image_bytes=image_bytes,
                    image_filename=img_path.name,
                    preds=preds,
                    model_name=model_meta["filename"]
                )
                
                # Customisation mineure du titre de métadonnées pour refléter le vrai manuscrit
                contract["metadata"]["document_type"] = ms["name"]
                
                # Étape 4 : Sauvegarde automatique du JSON final
                out_path = app.save_data_contract(contract)
                print(f" Sauvegardée -> `{out_path}`")
                
            except Exception as e:
                print(f" Erreur critique lors de la transcription : {e}")

if __name__ == "__main__":
    process_batch()
    print("\n🏁 Traitement par lot terminé. Les fichiers JSON sont dans `data/predictions/`.")