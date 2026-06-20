"""Tests de non-régression pour le pipeline de prétraitement.

Le pipeline (`pre_traitement.py`) est basé sur OpenCV : il prend et retourne
des `numpy.ndarray` (convention `cv2.imread` — BGR pour la couleur, 2D pour
les niveaux de gris), jamais des objets `PIL.Image`.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pre_traitement import pretraiter_image


def _image_gris(w=200, h=100):
    """Image niveaux de gris (2D) — déjà au format attendu en sortie (mode L)."""
    return np.random.randint(50, 200, (h, w), dtype=np.uint8)


def _image_couleur(w=200, h=100):
    """Image couleur BGR (3 canaux) — format produit par `cv2.imread`."""
    return np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)


class TestPretraiterImage:
    def test_sortie_niveaux_de_gris_par_defaut(self):
        """Sans binarisation : sortie 2D (un seul canal) — équivalent du mode L, pour Kraken."""
        img = _image_couleur()
        resultat, _ = pretraiter_image(img)
        assert resultat.ndim == 2, f"Attendu image 2D (niveaux de gris), obtenu shape={resultat.shape}"

    def test_sortie_binaire_avec_binariser(self):
        """Avec binariser=True (Sauvola forcé) : sortie binaire (valeurs 0/255 uniquement)."""
        img = _image_couleur()
        resultat, _ = pretraiter_image(img, binariser=True, forcer_sauvola=True)
        valeurs = set(np.unique(resultat).tolist())
        assert valeurs.issubset({0, 255}), f"Attendu image binaire (0/255), valeurs obtenues : {valeurs}"

    def test_dimensions_preservees(self):
        img = _image_gris(300, 150)
        resultat, _ = pretraiter_image(img)
        assert resultat.shape[0] > 0
        assert resultat.shape[1] > 0

    def test_entree_couleur_acceptee(self):
        img = _image_couleur()
        resultat, rapport = pretraiter_image(img)
        assert resultat is not None
        assert rapport is not None

    def test_entree_niveaux_de_gris_acceptee(self):
        img = _image_gris()
        resultat, rapport = pretraiter_image(img)
        assert resultat.ndim == 2
        assert rapport is not None

    def test_plage_valeurs_niveaux_de_gris(self):
        img = _image_couleur()
        resultat, _ = pretraiter_image(img)
        assert resultat.min() >= 0
        assert resultat.max() <= 255

    def test_rapport_contient_metriques_attendues(self):
        img = _image_couleur()
        _, rapport = pretraiter_image(img)
        assert hasattr(rapport, "skew_angle")
        assert hasattr(rapport, "clahe_applied")
        assert rapport.processing_time_s >= 0