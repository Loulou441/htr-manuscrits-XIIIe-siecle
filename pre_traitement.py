"""
Pipeline de prétraitement des scans de manuscrits médiévaux

Modules couverts :
    1. Anatomie du scan         — propriétés du parchemin et de l'encre
    2. Correction géométrique   — deskewing (3 méthodes) + perspective
    3. Amélioration du contraste — CLAHE avec diagnostic d'uniformité
    4. Réduction du bruit       — filtre médian + filtre gaussien avec diagnostics

Chaque opération suit le même pattern :
    diagnostiquer() → décider() → appliquer()

Usage :
    # Traiter une image unique
    python pre_traitement.py image.jpg --output image_traitee.jpg

    # Traiter un dossier complet (ALTO XML + images)
    python pre_traitement.py ./data/dataset --output ./data/preprocessed

    # Mode diagnostic seul (sans modification)
    python pre_traitement.py image.jpg --diagnose-only
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import cv2
import numpy as np
import shutil

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Extensions d'image supportées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# 0. STRUCTURES DE DONNÉES

@dataclass
class DiagnosticResult:
    # Résultat d'un diagnostic avec décision et paramètres recommandés
    metric_name: str
    metric_value: float
    decision: str           # "skip" | "apply_light" | "apply" | "apply_strong" | "manual"
    params: dict
    note: str = ""


@dataclass
class PreprocessingReport:
    # Rapport complet pour une image traitée
    image_path: str
    skew_angle: float
    skew_corrected: bool
    clahe_applied: bool
    clahe_clip_limit: float
    median_filter_applied: bool
    median_ksize: int
    gaussian_filter_applied: bool
    gaussian_sigma: float
    sauvola_applied: bool
    sauvola_block_size: int
    processing_time_s: float

# 1. UTILITAIRES — CHARGEMENT ET CONVERSION

def charger_image(path: str | Path) -> np.ndarray:
    """
    Charge une image (couleur ou niveau de gris) depuis un fichier.
    Retourne un tableau NumPy BGR (OpenCV) ou niveaux de gris.

    Raises:
        FileNotFoundError : si le fichier n'existe pas.
        ValueError        : si l'image ne peut pas être décodée.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image introuvable : {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de décoder l'image : {path}")
    return img


def vers_gris(image: np.ndarray) -> np.ndarray:
    # Convertit une image BGR en niveaux de gris si elle ne l'est pas déjà
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binariser_otsu(image_gris: np.ndarray) -> np.ndarray:
    #Binarise une image en niveaux de gris par la méthode d'Otsu (1979).
    _, binaire = cv2.threshold(
        image_gris, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binaire

def sauvegarder_image(image: np.ndarray, path: str | Path) -> None:
    """Sauvegarde une image NumPy vers un fichier."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

# 2. CORRECTION GÉOMÉTRIQUE — DESKEWING

## Deskewing par analyse de profils de projection

def estimer_angle_par_projection(
    image_binaire: np.ndarray,
    plage_deg: tuple[float, float] = (-15.0, 15.0),
    pas_deg: float = 0.5,
) -> float:
    """
    Estime l'angle d'inclinaison en maximisant la variance de la projection
    horizontale des pixels d'encre (pixels noirs).
    """
    # Inverser : pixels d'encre → 255 pour la somme
    encre = cv2.bitwise_not(image_binaire)
    h, w = encre.shape
    cx, cy = w // 2, h // 2

    meilleur_angle = 0.0
    meilleure_variance = -1.0

    angles = np.arange(plage_deg[0], plage_deg[1] + pas_deg, pas_deg)
    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotee = cv2.warpAffine(encre, M, (w, h), flags=cv2.INTER_NEAREST)
        # Projection horizontale : somme par ligne
        projection = rotee.sum(axis=1).astype(float)
        variance = float(np.var(projection))
        if variance > meilleure_variance:
            meilleure_variance = variance
            meilleur_angle = angle

    return meilleur_angle


## Deskewing par transformée de Hough

def estimer_angle_par_hough(image_binaire: np.ndarray) -> float:
    # Estime l'angle d'inclinaison dominant par transformée de Hough

    # Détection de contours (Canny)
    contours = cv2.Canny(image_binaire, 50, 150, apertureSize=3)

    # Transformée de Hough
    lignes = cv2.HoughLines(contours, 1, np.pi / 180, threshold=100)
    if lignes is None or len(lignes) == 0:
        log.warning("Hough : aucune ligne détectée, angle = 0°")
        return 0.0

    # Extraire les angles θ et les ramener dans [-90°, 90°]
    thetas = []
    for ligne in lignes:
        theta = ligne[0][1]
        angle_deg = np.degrees(theta) - 90.0
        # Filtrer les angles très proches de 90° (lignes verticales)
        if abs(angle_deg) < 45:
            thetas.append(angle_deg)

    if not thetas:
        return 0.0

    # L'angle dominant est la médiane (robuste aux outliers)
    return float(np.median(thetas))


## Deskewing par analyse fréquentielle (FFT)

def estimer_angle_par_fft(image_binaire: np.ndarray) -> float:
    # Estime l'angle d'inclinaison via la Transformée de Fourier rapide.

    # Calculer la FFT et son spectre de magnitude centré
    fft = np.fft.fft2(image_binaire.astype(float))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    # Normaliser en uint8 pour Hough
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Détecter l'orientation dominante par Hough dans le spectre
    lignes = cv2.HoughLines(magnitude_norm, 1, np.pi / 180, threshold=30)
    if lignes is None or len(lignes) == 0:
        return 0.0

    thetas = [np.degrees(l[0][1]) for l in lignes]
    # L'orientation dans le spectre est perpendiculaire → rotation de 90°
    angle_spectre = float(np.median(thetas))
    return angle_spectre - 90.0


# Diagnostic d'inclinaison

def diagnostiquer_inclinaison(
    image_gris: np.ndarray,
    methode: str = "projection",
) -> DiagnosticResult:
    # Diagnostique l'inclinaison d'une image et recommande une action
    
    ## Analyse recentrée sur le bloc de texte (Crop des 60% centraux)
    h, w = image_gris.shape
    zone_centrale = image_gris[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    
    ## Passage à un seuillage adaptatif local au lieu d'Otsu global
    binaire_local = cv2.adaptiveThreshold(
        zone_centrale, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 10
    )
    
    ## Respect de la méthode demandée en paramètre
    if methode == "projection":
        angle = estimer_angle_par_projection(binaire_local)
    elif methode == "hough":
        angle = estimer_angle_par_hough(binaire_local)
    elif methode == "fft":
        angle = estimer_angle_par_fft(binaire_local)
    else:
        log.warning("Méthode de deskewing inconnue : '%s'. Repli sur 'projection'.", methode)
        angle = estimer_angle_par_projection(binaire_local)
        
    abs_angle = abs(angle)

    if abs_angle < 0.3:
        decision = "skip"
        note = "Inclinaison négligeable — aucune correction"
    elif abs_angle <= 10.0:
        decision = "apply"
        note = f"Inclinaison modérée ({angle:.2f}°) — correction recommandée"
    else:
        decision = "manual"
        note = (
            f"Inclinaison > 10° ({angle:.2f}°) — probable erreur de détection "
            "ou page très penchée — vérifier manuellement"
        )

    return DiagnosticResult(
        metric_name="skew_angle_deg",
        metric_value=angle,
        decision=decision,
        params={"angle": angle},
        note=note,
    )


# Application de la rotation

def corriger_inclinaison(
    image: np.ndarray,
    angle: float,
    fond: int = 255,
) -> np.ndarray:
    # Applique une rotation pour corriger l'inclinaison (§2.2).
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    border_mode = cv2.BORDER_CONSTANT
    if len(image.shape) == 3:
        border_value = (fond, fond, fond)
    else:
        border_value = fond

    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
        borderValue=border_value,
    )


# Deskewing complet (diagnostic + correction)

def deskewer(
    image: np.ndarray,
    methode: str = "projection",
    forcer: bool = False,
) -> tuple[np.ndarray, DiagnosticResult]:
    # Pipeline deskewing complet : diagnostic → décision → correction.
    image_gris = vers_gris(image)
    diag = diagnostiquer_inclinaison(image_gris, methode=methode)

    if diag.decision == "skip" and not forcer:
        log.debug("Deskewing ignoré : angle=%.2f°", diag.metric_value)
        return image, diag

    if diag.decision == "manual" and not forcer:
        log.warning("Deskewing : %s", diag.note)
        return image, diag

    log.debug("Deskewing appliqué : angle=%.2f°", diag.metric_value)
    image_corrigee = corriger_inclinaison(image, diag.metric_value)
    return image_corrigee, diag


# Correction de perspective

def corriger_perspective(
    image: np.ndarray,
    coins_source: np.ndarray,
    largeur_cible: Optional[int] = None,
    hauteur_cible: Optional[int] = None,
) -> np.ndarray:
    # Corrige la distorsion de perspective par homographie.

    # Calculer la taille cible si non fournie
    if largeur_cible is None or hauteur_cible is None:
        tl, tr, br, bl = coins_source
        largeur = int(max(
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - bl),
        ))
        hauteur = int(max(
            np.linalg.norm(bl - tl),
            np.linalg.norm(br - tr),
        ))
        if largeur_cible is None:
            largeur_cible = largeur
        if hauteur_cible is None:
            hauteur_cible = hauteur

    coins_cible = np.array([
        [0, 0],
        [largeur_cible - 1, 0],
        [largeur_cible - 1, hauteur_cible - 1],
        [0, hauteur_cible - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        coins_source.astype(np.float32),
        coins_cible,
    )
    corrigee = cv2.warpPerspective(image, H, (largeur_cible, hauteur_cible))
    return corrigee

# 3. AMÉLIORATION DU CONTRASTE — CLAHE

def diagnostiquer_contraste(image_gris: np.ndarray) -> DiagnosticResult:
    # Diagnostique l'uniformité du contraste et décide d'appliquer CLAHE.
    p75 = np.percentile(image_gris, 75)
    pixels_fond = image_gris[image_gris > p75]

    if len(pixels_fond) == 0:
        std_fond = 0.0
    else:
        std_fond = float(np.std(pixels_fond))

    uniformite = 1.0 - std_fond / 255.0

    if uniformite < 0.4:
        decision = "apply"
        params = {"clip_limit": 3.0, "tile_grid_size": (8, 8)}
        note = f"Fond très hétérogène (uniformité={uniformite:.2f}) — CLAHE nécessaire"
    elif uniformite < 0.7:
        decision = "apply_light"
        params = {"clip_limit": 2.0, "tile_grid_size": (8, 8)}
        note = f"Fond modérément hétérogène (uniformité={uniformite:.2f}) — CLAHE léger"
    else:
        decision = "skip"
        params = {}
        note = f"Fond uniforme (uniformité={uniformite:.2f}) — CLAHE inutile"

    return DiagnosticResult(
        metric_name="uniformite_fond",
        metric_value=uniformite,
        decision=decision,
        params=params,
        note=note,
    )


def appliquer_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    # Applique CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        # Image en niveaux de gris directement
        return clahe.apply(image)

    # Image couleur : appliquer CLAHE uniquement sur le canal L (LAB)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def ameliorer_contraste(
    image: np.ndarray,
    forcer: bool = False,
) -> tuple[np.ndarray, DiagnosticResult]:
    # Pipeline CLAHE complet : diagnostic → décision → application.

    image_gris = vers_gris(image)
    diag = diagnostiquer_contraste(image_gris)

    if diag.decision == "skip" and not forcer:
        log.debug("CLAHE ignoré : %s", diag.note)
        return image, diag

    clip = diag.params.get("clip_limit", 2.0)
    grid = diag.params.get("tile_grid_size", (8, 8))
    log.debug("CLAHE appliqué : clipLimit=%.1f, grid=%s", clip, grid)
    image_traitee = appliquer_clahe(image, clip_limit=clip, tile_grid_size=grid)
    return image_traitee, diag

# 4. RÉDUCTION DU BRUIT

## Filtre médian — bruit sel-et-poivre

def diagnostiquer_bruit_sel_poivre(image_gris: np.ndarray) -> DiagnosticResult:
    # Diagnostique le bruit sel-et-poivre    
    total = image_gris.size
    extremes = int(np.sum((image_gris < 2) | (image_gris > 253)))
    fraction_pct = extremes / total * 100.0

    if fraction_pct < 0.1:
        decision = "skip"
        params = {}
        note = f"Bruit sel-et-poivre négligeable ({fraction_pct:.3f}%)"
    elif fraction_pct <= 1.0:
        decision = "apply"
        params = {"ksize": 3}
        note = f"Bruit léger ({fraction_pct:.3f}%) — filtre médian ksize=3"
    else:
        decision = "apply_strong"
        params = {"ksize": 5}
        note = (
            f"Bruit important ({fraction_pct:.3f}%) — filtre médian ksize=5 "
            "(vérifier la préservation des déliés)"
        )

    return DiagnosticResult(
        metric_name="fraction_pixels_extremes_pct",
        metric_value=fraction_pct,
        decision=decision,
        params=params,
        note=note,
    )


def filtrer_bruit_sel_poivre(
    image: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    # Applique un filtre médian pour éliminer le bruit sel-et-poivre

    if ksize % 2 == 0:
        raise ValueError(f"ksize doit être impair, reçu : {ksize}")
    return cv2.medianBlur(image, ksize)


def reduire_bruit_sel_poivre(
    image: np.ndarray,
    forcer: bool = False,
    ksize_override: Optional[int] = None,
) -> tuple[np.ndarray, DiagnosticResult]:
    # Pipeline filtre médian complet : diagnostic → décision → application.

    image_gris = vers_gris(image)
    diag = diagnostiquer_bruit_sel_poivre(image_gris)

    if diag.decision == "skip" and not forcer and ksize_override is None:
        log.debug("Filtre médian ignoré : %s", diag.note)
        return image, diag

    ksize = ksize_override or diag.params.get("ksize", 3)
    log.debug("Filtre médian appliqué : ksize=%d", ksize)
    image_traitee = filtrer_bruit_sel_poivre(image, ksize=ksize)
    diag.params["ksize_used"] = ksize
    return image_traitee, diag


## Filtre gaussien — bruit de fond

def diagnostiquer_bruit_gaussien(image_gris: np.ndarray) -> DiagnosticResult:
    # Diagnostique le bruit gaussien (bruit de fond) (§4.2.B).
    p75 = np.percentile(image_gris, 75)
    pixels_fond = image_gris[image_gris > p75]

    if len(pixels_fond) == 0:
        sigma_fond = 0.0
    else:
        sigma_fond = float(np.std(pixels_fond))

    if sigma_fond < 5.0:
        decision = "skip"
        params = {}
        note = f"Bruit gaussien négligeable (σ_fond={sigma_fond:.1f})"
    elif sigma_fond <= 15.0:
        decision = "apply"
        params = {"sigma": 0.8}
        note = f"Bruit modéré (σ_fond={sigma_fond:.1f}) — sigma=0.8"
    else:
        decision = "apply_strong"
        params = {"sigma": 1.2}
        note = (
            f"Bruit élevé (σ_fond={sigma_fond:.1f}) — sigma=1.2 "
            "(vérifier absence de flou sur les lettres)"
        )

    return DiagnosticResult(
        metric_name="sigma_fond",
        metric_value=sigma_fond,
        decision=decision,
        params=params,
        note=note,
    )


def filtrer_bruit_gaussien(
    image: np.ndarray,
    sigma: float = 0.8,
) -> np.ndarray:
    # Applique un filtre gaussien pour atténuer le bruit de fond (§4.2.A).

    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def reduire_bruit_gaussien(
    image: np.ndarray,
    forcer: bool = False,
    sigma_override: Optional[float] = None,
) -> tuple[np.ndarray, DiagnosticResult]:
    # Pipeline filtre gaussien complet : diagnostic → décision → application.

    image_gris = vers_gris(image)
    diag = diagnostiquer_bruit_gaussien(image_gris)

    if diag.decision == "skip" and not forcer and sigma_override is None:
        log.debug("Filtre gaussien ignoré : %s", diag.note)
        return image, diag

    sigma = sigma_override or diag.params.get("sigma", 0.8)
    log.debug("Filtre gaussien appliqué : sigma=%.2f", sigma)
    image_traitee = filtrer_bruit_gaussien(image, sigma=sigma)
    diag.params["sigma_used"] = sigma
    return image_traitee, diag

# 5. SÉLECTION AUTOMATIQUE DE LA MÉTHODE PAR TYPE D'ÉCRITURE

# Table de correspondance : mots-clés dans le nom du script → méthode optimale.
#
# Justification des choix :
#   projection — Gothic Textualis (Formata, Libraria, Southern) :
#       Les hastes et jambages verticaux très marqués créent dans le spectre FFT
#       des fréquences verticales plus énergétiques que les lignes horizontales.
#       La FFT se trompe d'orientation. Hough est perturbé par les réglures à la
#       mine de plomb omniprésentes dans les manuscrits parisiens du XIIIe.
#       La projection reste la méthode la plus stable sur ce type.
#
#   hough    — Semitextualis Currens, Textualis Currens :
#       Écriture plus cursive, traits moins verticaux, lignes de base plus nettes.
#       Hough détecte bien l'orientation dominante.
#
#   fft      — Non utilisé par défaut. Disponible manuellement si les scans
#       sont très propres et l'écriture très régulière.

SCRIPT_TO_METHODE: dict[str, str] = {
    # Gothic Textualis et variantes → projection
    "gothic textualis":          "projection",
    "textualis formata":         "projection",
    "textualis libraria":        "projection",
    "southern textualis":        "projection",
    "s. textualis":              "projection",
    "caroline":                  "projection",
    "praegothica":               "projection",
    # Cursive et semi-cursive → hough
    "semitextualis currens":     "hough",
    "textualis currens":         "hough",
    "cursiva":                   "hough",
    "hybrida":                   "hough",
    "bastarda":                  "hough",
}

METHODE_DEFAUT = "projection"

def methode_pour_script(script: str) -> str:
    # Retourne la méthode de deskewing optimale pour un type d'écriture donné.
    # Insensible à la casse, tolère les variantes de nommage.
    script_lower = script.lower().strip()
    for motcle, methode in SCRIPT_TO_METHODE.items():
        if motcle in script_lower:
            return methode
    log.warning(
        "Script non reconnu : '%s' — méthode par défaut utilisée ('%s')",
        script, METHODE_DEFAUT,
    )
    return METHODE_DEFAUT


def charger_manifest(dataset_dir: Path) -> dict[str, str]:
    # Lit manifest.json produit par dataset.py et construit un dict slug_dossier → méthode_deskew optimale.
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        log.warning(
            "manifest.json introuvable dans %s — méthode uniforme '%s' utilisée.",
            dataset_dir, METHODE_DEFAUT,
        )
        return {}

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, str] = {}
    for mss in data.get("manuscripts", []):
        shelfmark = mss.get("shelfmark", "")
        script    = mss.get("script", "")
        slug      = shelfmark.replace(" ", "_").replace("/", "-").replace(".", "")
        methode   = methode_pour_script(script)
        mapping[slug] = methode
        log.debug("  %-35s script=%-35s → %s", slug, script, methode)

    log.info(
        "Manifest chargé : %d manuscrits — méthodes : %s",
        len(mapping),
        {m: sum(1 for v in mapping.values() if v == m) for m in set(mapping.values())},
    )
    return mapping


def methode_pour_image(
    image_path: Path,
    dataset_dir: Path,
    manifest_mapping: dict[str, str],
) -> str:
    #Détermine la méthode de deskewing pour une image en remontant l'arborescence pour trouver le slug du manuscrit (2e niveau sous dataset_dir).
    if not manifest_mapping:
        return METHODE_DEFAUT
    try:
        parties = image_path.relative_to(dataset_dir).parts
        if len(parties) >= 2:
            slug = parties[1]
            if slug in manifest_mapping:
                return manifest_mapping[slug]
            # Correspondance insensible à la casse
            for key, methode in manifest_mapping.items():
                if key.lower() == slug.lower():
                    return methode
    except ValueError:
        pass
    log.debug("Slug non trouvé pour %s — méthode par défaut", image_path.name)
    return METHODE_DEFAUT

# 6. PIPELINE COMPLET

def pretraiter_image(
    image: np.ndarray,
    methode_deskew: str = "projection",
    forcer_deskew: bool = False,
    forcer_clahe: bool = False,
    forcer_median: bool = False,
    forcer_gaussien: bool = False,
) -> tuple[np.ndarray, PreprocessingReport]:
    # Pipeline de prétraitement complet pour un scan de manuscrit médiéval.

    t0 = time.perf_counter()
    img = image.copy()

    # Étape 1 : Deskewing
    img, diag_skew = deskewer(img, methode=methode_deskew, forcer=forcer_deskew)
    skew_corrected = diag_skew.decision in ("apply", "apply_strong", "manual") or forcer_deskew

    # Étape 2 : CLAHE
    img, diag_clahe = ameliorer_contraste(img, forcer=forcer_clahe)
    clahe_applied = diag_clahe.decision in ("apply", "apply_light") or forcer_clahe
    clahe_clip = diag_clahe.params.get("clip_limit", 0.0)

    # Étape 3a : Filtre médian
    img, diag_median = reduire_bruit_sel_poivre(img, forcer=forcer_median)
    median_applied = diag_median.decision in ("apply", "apply_strong") or forcer_median
    median_ksize = diag_median.params.get("ksize_used", diag_median.params.get("ksize", 0))

    # Étape 3b : Filtre gaussien
    img, diag_gauss = reduire_bruit_gaussien(img, forcer=forcer_gaussien)
    gauss_applied = diag_gauss.decision in ("apply", "apply_strong") or forcer_gaussien
    gauss_sigma = diag_gauss.params.get("sigma_used", diag_gauss.params.get("sigma", 0.0))

    elapsed = time.perf_counter() - t0

    rapport = PreprocessingReport(
        image_path="",
        skew_angle=diag_skew.metric_value,
        skew_corrected=skew_corrected,
        clahe_applied=clahe_applied,
        clahe_clip_limit=clahe_clip,
        median_filter_applied=median_applied,
        median_ksize=median_ksize,
        gaussian_filter_applied=gauss_applied,
        gaussian_sigma=gauss_sigma,
        processing_time_s=elapsed,
    )

    return img, rapport


def pretraiter_fichier(
    input_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> PreprocessingReport:
    # Charge une image, applique le pipeline complet, sauvegarde le résultat.

    image = charger_image(input_path)
    image_traitee, rapport = pretraiter_image(image, **kwargs)
    rapport.image_path = str(input_path)
    sauvegarder_image(image_traitee, output_path)
    return rapport


def pretraiter_dossier(
    input_dir: str | Path,
    output_dir: str | Path,
    extensions: set[str] = IMAGE_EXTENSIONS,
    auto_methode: bool = True,
    **kwargs,
) -> list[PreprocessingReport]:
    # Applique le pipeline à toutes les images d'un dossier (récursif).

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rapports = []

    # Charger le manifest pour la sélection automatique de méthode
    manifest_mapping: dict[str, str] = {}
    if auto_methode:
        manifest_mapping = charger_manifest(input_dir)

    fichiers = [
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    ]

    log.info("Prétraitement de %d images dans %s ...", len(fichiers), input_dir)
    if auto_methode and manifest_mapping:
        log.info("Sélection automatique de méthode activée (lecture du manifest).")
    elif auto_methode:
        log.info("Manifest absent — méthode uniforme '%s' appliquée.", METHODE_DEFAUT)

    for i, fichier in enumerate(sorted(fichiers), 1):
        relatif = fichier.relative_to(input_dir)
        dest = output_dir / relatif
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Sélection de la méthode pour cette image
        if auto_methode:
            methode = methode_pour_image(fichier, input_dir, manifest_mapping)
            kw = {**kwargs, "methode_deskew": methode}
        else:
            kw = kwargs

        log.info(
            "[%d/%d] %s  [deskew=%s]",
            i, len(fichiers), relatif,
            kw.get("methode_deskew", METHODE_DEFAUT),
        )
        try:
            rapport = pretraiter_fichier(fichier, dest, **kw)
            rapports.append(rapport)
        except Exception as e:
            log.error("Erreur sur %s : %s", fichier, e)

    # Copier les fichiers XML (ALTO) sans modification
    xml_files = list(input_dir.rglob("*.xml"))
    for xml in xml_files:
        dest_xml = output_dir / xml.relative_to(input_dir)
        dest_xml.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(xml, dest_xml)

    log.info(
        "Terminé : %d images traitées, %d XML copiés.",
        len(rapports), len(xml_files),
    )
    return rapports


def afficher_rapport(rapport: PreprocessingReport) -> None:
    """Affiche un rapport de prétraitement lisible."""
    print(f"\n── Rapport prétraitement : {Path(rapport.image_path).name}")
    print(f"   Deskewing  : {'✓' if rapport.skew_corrected else '–'}  "
          f"angle={rapport.skew_angle:+.2f}°")
    print(f"   CLAHE      : {'✓' if rapport.clahe_applied else '–'}  "
          f"clipLimit={rapport.clahe_clip_limit:.1f}")
    print(f"   Médian     : {'✓' if rapport.median_filter_applied else '–'}  "
          f"ksize={rapport.median_ksize}")
    print(f"   Gaussien   : {'✓' if rapport.gaussian_filter_applied else '–'}  "
          f"sigma={rapport.gaussian_sigma:.2f}")
    print(f"   Durée      : {rapport.processing_time_s*1000:.1f} ms")

# 6. INTERFACE EN LIGNE DE COMMANDE

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de prétraitement des scans de manuscrits médiévaux."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("./data/dataset"),
        help="Image source ou dossier à traiter (défaut : ./data/dataset)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Image/dossier de sortie (défaut : ./data/preprocessed si input=dataset, sinon <input>_preprocessed)",
    )
    parser.add_argument(
        "--methode-deskew",
        choices=["projection", "fft", "hough"],
        default="projection",
        help="Méthode de deskewing utilisée si --no-auto-methode (défaut : projection)",
    )
    parser.add_argument(
        "--no-auto-methode",
        action="store_true",
        help=(
            "Désactive la sélection automatique par script paléographique. "
            "Applique --methode-deskew uniformément à toutes les images."
        ),
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Diagnostic uniquement — aucune modification enregistrée",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Forcer l'application de toutes les corrections sans diagnostic",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Active les messages DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = args.input
    # Dossier de sortie par défaut : si input est le dataset par défaut → ./data/preprocessed
    DEFAULT_DATASET = Path("./data/dataset")
    if args.output is None:
        if input_path == DEFAULT_DATASET or input_path.resolve() == DEFAULT_DATASET.resolve():
            output_path = Path("./data/preprocessed")
        elif input_path.is_dir():
            output_path = input_path.parent / f"{input_path.name}_preprocessed"
        else:
            output_path = input_path.with_name(
                input_path.stem + "_preprocessed" + input_path.suffix
            )
    else:
        output_path = args.output

    kwargs = {
        "methode_deskew": args.methode_deskew,
        "forcer_deskew": args.force_all,
        "forcer_clahe": args.force_all,
        "forcer_median": args.force_all,
        "forcer_gaussien": args.force_all,
    }

    if input_path.is_dir():
        rapports = pretraiter_dossier(
            input_path, output_path,
            auto_methode=not args.no_auto_methode,
            **kwargs,
        )
        for r in rapports:
            if args.diagnose_only or args.verbose:
                afficher_rapport(r)
    else:
        if args.diagnose_only:
            image = charger_image(input_path)
            _, rapport = pretraiter_image(image, **kwargs)
            rapport.image_path = str(input_path)
            afficher_rapport(rapport)
        else:
            rapport = pretraiter_fichier(input_path, output_path, **kwargs)
            afficher_rapport(rapport)
            log.info("Résultat sauvegardé dans %s", output_path)


if __name__ == "__main__":
    main()
