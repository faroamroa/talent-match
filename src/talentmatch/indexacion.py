"""Convierte las extracciones en textos para embeber y payloads para filtrar, y las indexa en Qdrant."""

import argparse
import json
import re
import unicodedata
import uuid
from datetime import date

from qdrant_client import QdrantClient, models

from talentmatch.config import CARPETA_EXTRACCIONES, RAIZ
from talentmatch.embeddings import MODELOS, embeber, recortar

CARPETA_QDRANT = RAIZ / "data" / "qdrant"
SECCIONES = ("perfil", "skills", "educacion")
SIN_INFO = "sin información"

ORDEN_NIVEL = {
    "bachiller": 1,
    "tecnico": 2,
    "tecnologo": 3,
    "pregrado": 4,
    "especializacion": 5,
    "maestria": 6,
    "doctorado": 7,
}


# Carga
def cargar(tipo: str) -> dict[str, dict]:
    """archivo original (sin .json) -> datos extraídos."""
    return {
        f.name.removesuffix(".json"): json.loads(f.read_text(encoding="utf-8"))["datos"]
        for f in sorted((CARPETA_EXTRACCIONES / tipo).glob("*.json"))
    }


# Normalización y campos derivados
def normalizar(texto: str | None) -> str | None:
    """'Medellín, Antioquia' -> 'medellin'. Se aplica igual a hojas de vida y vacantes."""
    if not texto:
        return None
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.casefold().split(",")[0].strip() or None


def ciudades(texto: str | None) -> list[str]:
    """'Medellín - Manizales - Bogotá, D.C.' -> ['medellin', 'manizales', 'bogota']."""
    partes = re.split(r"\s+-\s+|/|;|\|", texto or "")
    return [c for c in (normalizar(p) for p in partes) if c]


def nivel_max(cv: dict) -> int | None:
    """Nivel educativo más alto como ordinal. Los estudios incompletos también cuentan."""
    niveles = [ORDEN_NIVEL.get(e["nivel"]) for e in cv["educacion"]]
    return max([n for n in niveles if n], default=None)


def _mes(fecha: dict | None, por_defecto: int) -> int | None:
    if not fecha or not fecha.get("anio"):
        return None
    return fecha["anio"] * 12 + (fecha.get("mes") or por_defecto) - 1


def anios_experiencia(cv: dict, hoy: date | None = None) -> float | None:
    """Años de experiencia sin contar dos veces los meses de trabajos simultáneos."""
    hoy = hoy or date.today()
    meses: set[int] = set()
    for e in cv["experiencia"]:
        inicio = _mes(e["fecha_inicio"], por_defecto=1)
        fin = _mes(e["fecha_fin"], por_defecto=12)
        if fin is None and e["es_actual"]:
            fin = hoy.year * 12 + hoy.month - 1
        if inicio is None or fin is None or fin < inicio:
            continue  # fechas desconocidas: no se adivina
        meses.update(range(inicio, fin + 1))
    return round(len(meses) / 12, 1) if meses else None


# Textos por sección
def unir(*partes: str | list[str] | None) -> str:
    planas = []
    for p in partes:
        planas.extend(p if isinstance(p, list) else [p])
    return ". ".join(p.strip() for p in planas if p and p.strip())


def textos_cv(cv: dict) -> dict[str, str]:
    experiencia = [f"{e['cargo'] or ''}: {e['descripcion'] or ''}" for e in cv["experiencia"]]
    return {
        "perfil": unir(cv["titular"], cv["resumen_perfil"], experiencia),
        "skills": unir(", ".join(cv["herramientas"] + cv["habilidades"])),
        "educacion": unir(
            [e["titulo"] for e in cv["educacion"]],
            [c["nombre"] for c in cv["certificaciones"]],
        ),
    }


def textos_vacante(v: dict) -> dict[str, str]:
    herramientas = v["herramientas_obligatorias"] + v["herramientas_deseables"]
    habilidades = v["habilidades_obligatorias"] + v["habilidades_deseables"]
    return {
        "perfil": unir(
            v["titulo"],
            v["descripcion_cargo"],
            v["responsabilidades"],
            [e["area_o_cargo"] for e in v["experiencia_especifica"]],
        ),
        "skills": unir(", ".join(herramientas + habilidades)),
        "educacion": unir(", ".join(v["profesiones_requeridas"]), v["certificaciones_requeridas"]),
    }


def payload_cv(archivo: str, cv: dict) -> dict:
    residencia = cv["residencia"] or {}
    return {
        "archivo": archivo,
        "titular": cv["titular"],
        "ciudad": normalizar(residencia.get("ciudad")),
        "nivel_max": nivel_max(cv),
        "anios_exp": anios_experiencia(cv),
    }


# Qdrant
def conectar(en_memoria: bool = False) -> QdrantClient:
    return QdrantClient(":memory:") if en_memoria else QdrantClient(path=str(CARPETA_QDRANT))


def coleccion(alias: str) -> str:
    return f"cvs_{alias}"


def nombre_vector(seccion: str, dim: int) -> str:
    return f"{seccion}_{dim}"


def indexar(cliente: QdrantClient, alias: str, cvs: dict[str, dict]) -> int:
    """Una colección por modelo, con un vector por cada (sección, tamaño)."""
    dims = MODELOS[alias].dims
    nombre = coleccion(alias)
    if cliente.collection_exists(nombre):
        cliente.delete_collection(nombre)
    cliente.create_collection(
        nombre,
        vectors_config={
            nombre_vector(s, d): models.VectorParams(size=d, distance=models.Distance.COSINE)
            for s in SECCIONES
            for d in dims
        },
    )

    textos = [textos_cv(cv) for cv in cvs.values()]
    completos = {s: embeber(alias, [t[s] or SIN_INFO for t in textos], "documento") for s in SECCIONES}

    puntos = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, archivo)),
            vector={
                nombre_vector(s, d): recortar(completos[s][i], d) for s in SECCIONES for d in dims
            },
            payload=payload_cv(archivo, cv),
        )
        for i, (archivo, cv) in enumerate(cvs.items())
    ]
    cliente.upsert(nombre, puntos)
    return len(puntos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embebe las hojas de vida y las guarda en Qdrant")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS, default=list(MODELOS))
    args = parser.parse_args()

    cvs = cargar("cvs")
    cliente = conectar()
    for alias in args.modelos:
        print(f"{alias}: {indexar(cliente, alias, cvs)} hojas de vida en '{coleccion(alias)}'")
    cliente.close()
