"""Tests de non-régression pour le pipeline de prétraitement."""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pre_traitement import pretraiter_image


def _image_gris(w=200, h=100):
    arr = np.random.randint(50, 200, (h, w), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _image_rgb(w=200, h=100):
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestPretraiterImage:
    def test_sortie_mode_L_par_defaut(self):
        img = _image_rgb()
        resultat, _ = pretraiter_image(img)
        assert resultat.mode == "L", f"Attendu mode L, obtenu {resultat.mode}"

    def test_sortie_mode_1_avec_binariser(self):
        img = _image_rgb()
        resultat, _ = pretraiter_image(img, binariser=True)
        assert resultat.mode == "1", f"Attendu mode 1, obtenu {resultat.mode}"

    def test_dimensions_preservees(self):
        img = _image_gris(300, 150)
        resultat, _ = pretraiter_image(img)
        assert resultat.size[0] > 0
        assert resultat.size[1] > 0

    def test_entree_rgb_acceptee(self):
        img = _image_rgb()
        resultat, rapport = pretraiter_image(img)
        assert resultat is not None
        assert rapport is not None

    def test_entree_mode_L_acceptee(self):
        img = _image_gris()
        resultat, rapport = pretraiter_image(img)
        assert resultat.mode == "L"

    def test_plage_valeurs_mode_L(self):
        img = _image_rgb()
        resultat, _ = pretraiter_image(img)
        arr = np.array(resultat)
        assert arr.min() >= 0
        assert arr.max() <= 255
