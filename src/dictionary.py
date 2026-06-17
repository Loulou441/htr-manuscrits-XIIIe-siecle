import os
import tarfile
import json
import requests
from pystardict import Dictionary

# --- CONFIGURATION ---
# Utilisation des liens "raw" de GitHub pour télécharger directement les fichiers
URL_STARDICT_TAR = "https://github.com/Vuizur/Wiktionary-Dictionaries/raw/master/Old%20French-English%20Wiktionary%20dictionary%20stardict.tar.gz"
URL_CLTK_TXT = "https://raw.githubusercontent.com/cltk/french_lexicon_cltk/master/docs/lexiquedelancienfrancais.txt"

DIR_DATA = "dictionnaire_data"
FILE_TAR = os.path.join(DIR_DATA, "old_french_stardict.tar.gz")
FILE_CLTK = os.path.join(DIR_DATA, "lexiquedelancienfrancais.txt")
OUTPUT_JSON = "dictionnaire_ancien_francais.json"

# Création du dossier temporaire
os.makedirs(DIR_DATA, exist_ok=True)

def download_file(url, destination):
    """Télécharge un fichier depuis une URL."""
    print(f"Téléchargement de {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Enregistré : {destination}")
    else:
        raise Exception(f"Impossible de télécharger le fichier. Code statut : {response.status_code}")

# --- 1. TÉLÉCHARGEMENT ET EXTRACTION ---
download_file(URL_STARDICT_TAR, FILE_TAR)
download_file(URL_CLTK_TXT, FILE_CLTK)

print("Extraction de l'archive StarDict...")
with tarfile.open(FILE_TAR, "r:gz") as tar:
    tar.extractall(path=DIR_DATA)

# Trouver le chemin du dictionnaire extrait (recherche du fichier .ifo)
dict_dir = None
for root, dirs, files in os.walk(DIR_DATA):
    if any(f.endswith('.ifo') for f in files):
        # Obtenir le nom de base sans extension
        ifo_file = [f for f in files if f.endswith('.ifo')][0]
        dict_dir = os.path.join(root, ifo_file[:-4])
        break

if not dict_dir:
    raise FileNotFoundError("Impossible de trouver les fichiers du dictionnaire StarDict extraits.")

# --- 2. PARSING DES DICTIONNAIRES ---
dictionnaire_complet = {}

def ajouter_entree(mot, definition, source):
    """Ajoute proprement une définition au dictionnaire unifié."""
    mot = mot.strip().lower() # Normalisation de base
    if not mot:
        return
        
    if mot not in dictionnaire_complet:
        dictionnaire_complet[mot] = {
            "wiktionary_en": [],
            "cltk_fr": []
        }
    
    # Éviter les doublons de définitions
    if definition not in dictionnaire_complet[mot][source]:
        dictionnaire_complet[mot][source].append(definition)

# --- Source A : StarDict (Wiktionary Ancien Français -> Anglais) ---
print("Traitement du dictionnaire StarDict (Wiktionary)...")
try:
    # pystardict charge le dictionnaire via le préfixe du fichier
    dict_stardict = Dictionary(dict_dir)
    for word in dict_stardict.keys():
        # Décodage de la définition (souvent stockée en bytes)
        definition = dict_stardict[word]
        if isinstance(definition, bytes):
            definition = definition.decode('utf-8', errors='ignore')
        
        ajouter_entree(word, definition.strip(), "wiktionary_en")
except Exception as e:
    print(f"Erreur lors de la lecture du StarDict : {e}")

# --- Source B : CLTK Text (Lexique de l'ancien français -> Français) ---
print("Traitement du lexique CLTK...")
with open(FILE_CLTK, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): # Sauter les lignes vides ou commentaires s'il y en a
            continue
        
        # Le format du fichier CLTK est généralement : "mot : définition" ou "mot , définition"
        # On sépare au premier séparateur trouvé
        if " : " in line:
            mot, definition = line.split(" : ", 1)
        elif "\t" in line:
            mot, definition = line.split("\t", 1)
        else:
            # Séparateur par espace si aucun autre (ajustable selon la structure exacte)
            parts = line.split(" ", 1)
            if len(parts) == 2:
                mot, definition = parts
            else:
                continue
                
        ajouter_entree(mot, definition.strip(), "cltk_fr")

# --- 3. SAUVEGARDE DU RÉSULTAT ---
print(f"Fusion terminée. Nombre total de mots uniques : {len(dictionnaire_complet)}")
print(f"Sauvegarde dans {OUTPUT_JSON}...")

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(dictionnaire_complet, f, ensure_ascii=False, indent=4)

print("Terminé avec succès !")