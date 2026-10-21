from pathlib import Path


def raiz_proyecto() -> Path:
    for carpeta in Path(__file__).resolve().parents:
        if (carpeta / "pyproject.toml").exists():
            return carpeta
    raise FileNotFoundError("No se encontró pyproject.toml")


RAIZ = raiz_proyecto()
RAW = RAIZ / "data" / "raw"
CARPETA_EXTRACCIONES = RAIZ / "data" / "processed" / "extracciones"