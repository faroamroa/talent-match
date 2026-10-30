"""Convierte las extracciones en textos para embeber y payloads para filtrar, y las indexa en Qdrant."""

import argparse
import json
import re
import unicodedata
import uuid
from datetime import date

from qdrant_client import QdrantClient, models

from talentmatch.config import CARPETA_EXTRACCIONES, RAIZ
from talentmatch.embeddings import MODELOS, TAMANO_LOTE, embeber, recortar

CARPETA_QDRANT = RAIZ / "data" / "qdrant"
SECCIONES = ("perfil", "skills", "educacion")  # por separado: solo para experimentar
SECCION_COMPLETA = "completo"  # todo el CV en un solo vector
# 1 vector rindió igual que 3 (mejor MRR, algo menos de R@10) con un tercio de la memoria
SECCIONES_PRODUCCION = (SECCION_COMPLETA,)
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

ORDEN_MCER = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6, "nativo": 7}

# Sinónimos (ya normalizados: sin tildes y en minúsculas) -> nombre canónico
IDIOMAS = {
    "ingles": "ingles", "english": "ingles",
    "espanol": "espanol", "spanish": "espanol", "castellano": "espanol",
    "portugues": "portugues", "portuguese": "portugues",
    "frances": "frances", "french": "frances",
    "aleman": "aleman", "german": "aleman",
    "italiano": "italiano", "italian": "italiano",
}


# Carga
def cargar(tipo: str) -> dict[str, dict]:
    """archivo original (sin .json) -> datos extraídos."""
    return {
        f.name.removesuffix(".json"): json.loads(f.read_text(encoding="utf-8"))["datos"]
        for f in sorted((CARPETA_EXTRACCIONES / tipo).glob("*.json"))
    }


# Normalización y campos derivados
def sin_tildes(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().casefold()


def normalizar(texto: str | None) -> str | None:
    """'Medellín, Antioquia' -> 'medellin'. Se aplica igual a hojas de vida y vacantes."""
    if not texto:
        return None
    return sin_tildes(texto).split(",")[0].strip() or None


def ciudades(texto: str | None) -> list[str]:
    """'Medellín - Manizales - Bogotá, D.C.' -> ['medellin', 'manizales', 'bogota']."""
    partes = re.split(r"\s+-\s+|/|;|\|", texto or "")
    return [c for c in (normalizar(p) for p in partes) if c]


def nivel_max(cv: dict) -> int | None:
    """Nivel educativo más alto como ordinal. Los estudios incompletos también cuentan."""
    niveles = [ORDEN_NIVEL.get(e["nivel"]) for e in cv["educacion"]]
    return max([n for n in niveles if n], default=None)


def nivel_mcer(idioma: dict) -> int | None:
    """Ordinal MCER del campo extraído o, si falta, de lo escrito: 'B2+ or C1' -> 5, 'Native' -> 7."""
    texto = sin_tildes(f"{idioma.get('nivel_mcer') or ''} {idioma.get('nivel_texto') or ''}")
    niveles = [ORDEN_MCER[n] for n in re.findall(r"\b([abc][12])\b", texto)]
    if re.search(r"nativ|native|materna", texto):
        niveles.append(ORDEN_MCER["nativo"])
    return max(niveles, default=None)


def idiomas(lista: list[dict]) -> list[dict]:
    """[{'idioma': 'English', 'nivel_texto': 'C1'}] -> [{'idioma': 'ingles', 'mcer': 5}]; une repetidos."""
    mejores: dict[str, int | None] = {}
    for i in lista:
        nombre = normalizar(i.get("idioma"))
        if not nombre:
            continue
        canonico = IDIOMAS.get(nombre, nombre)
        niveles = [n for n in (mejores.get(canonico), nivel_mcer(i)) if n]
        mejores[canonico] = max(niveles, default=None)
    return [{"idioma": k, "mcer": v} for k, v in mejores.items()]


def _mes(fecha: dict | None, por_defecto: int) -> int | None:
    if not fecha or not fecha.get("anio"):
        return None
    return fecha["anio"] * 12 + (fecha.get("mes") or por_defecto) - 1


def meses_experiencia(e: dict, hoy: date | None = None) -> tuple[int, int] | None:
    """(mes inicial, mes final) de un cargo como índices absolutos; None si las fechas no alcanzan."""
    hoy = hoy or date.today()
    inicio = _mes(e["fecha_inicio"], por_defecto=1)
    fin = _mes(e["fecha_fin"], por_defecto=12)
    if fin is None and e["es_actual"]:
        fin = hoy.year * 12 + hoy.month - 1
    if inicio is None or fin is None or fin < inicio:
        return None  # fechas desconocidas: no se adivina
    return inicio, fin


def anios(rangos: list[tuple[int, int]]) -> float:
    """Años cubiertos por varios rangos de meses, sin contar dos veces los meses simultáneos."""
    meses: set[int] = set()
    for inicio, fin in rangos:
        meses.update(range(inicio, fin + 1))
    return round(len(meses) / 12, 1)


def anios_experiencia(cv: dict, hoy: date | None = None) -> float | None:
    """Experiencia total. None (desconocida) si ningún cargo tiene fechas utilizables."""
    rangos = [r for e in cv["experiencia"] if (r := meses_experiencia(e, hoy))]
    return anios(rangos) if rangos else None


# Textos por sección
def unir(*partes: str | list[str] | None) -> str:
    planas = []
    for p in partes:
        planas.extend(p if isinstance(p, list) else [p])
    return ". ".join(p.strip() for p in planas if p and p.strip())


def _con_completo(textos: dict[str, str]) -> dict[str, str]:
    """Agrega 'completo': todas las secciones en un solo texto, para comparar 1 vector contra 3."""
    return textos | {SECCION_COMPLETA: unir(*textos.values())}


def textos_cv(cv: dict) -> dict[str, str]:
    experiencia = [f"{e['cargo'] or ''}: {e['descripcion'] or ''}" for e in cv["experiencia"]]
    return _con_completo({
        "perfil": unir(cv["titular"], cv["resumen_perfil"], experiencia),
        "skills": unir(", ".join(cv["herramientas"] + cv["habilidades"])),
        "educacion": unir(
            [e["titulo"] for e in cv["educacion"]],
            [c["nombre"] for c in cv["certificaciones"]],
        ),
    })


def textos_vacante(v: dict) -> dict[str, str]:
    herramientas = v["herramientas_obligatorias"] + v["herramientas_deseables"]
    habilidades = v["habilidades_obligatorias"] + v["habilidades_deseables"]
    areas = [e["area_o_cargo"] for e in v["experiencia_especifica"]]
    return _con_completo({
        "perfil": unir(v["titulo"], v["descripcion_cargo"], v["responsabilidades"], areas),
        "skills": unir(", ".join(herramientas + habilidades)),
        "educacion": unir(", ".join(v["profesiones_requeridas"]), v["certificaciones_requeridas"]),
    })


def payload_cv(archivo: str, cv: dict) -> dict:
    residencia = cv["residencia"] or {}
    return {
        "archivo": archivo,
        "titular": cv["titular"],
        "ciudad": normalizar(residencia.get("ciudad")),
        "nivel_max": nivel_max(cv),
        "anios_exp": anios_experiencia(cv),
        "idiomas": idiomas(cv["idiomas"]),
    }


# Qdrant
def conectar(en_memoria: bool = False) -> QdrantClient:
    return QdrantClient(":memory:") if en_memoria else QdrantClient(path=str(CARPETA_QDRANT))


def coleccion(alias: str) -> str:
    return f"cvs_{alias}"


def nombre_vector(seccion: str, dim: int) -> str:
    return f"{seccion}_{dim}"


def _recrear(cliente: QdrantClient, nombre: str, secciones: tuple[str, ...], dims: tuple[int, ...]) -> None:
    if cliente.collection_exists(nombre):
        cliente.delete_collection(nombre)
    cliente.create_collection(
        nombre,
        vectors_config={
            nombre_vector(s, d): models.VectorParams(size=d, distance=models.Distance.COSINE)
            for s in secciones
            for d in dims
        },
    )


def indexar(
    cliente: QdrantClient,
    alias: str,
    cvs: dict[str, dict],
    dims: tuple[int, ...] | None = None,
    secciones: tuple[str, ...] = SECCIONES_PRODUCCION,
) -> int:
    """Una colección por modelo, con un vector por cada (sección, tamaño).

    Por defecto, 1 vector ('completo') del tamaño de producción: cada sección o tamaño
    extra multiplica la memoria.
    """
    dims = dims or (MODELOS[alias].dim_produccion,)
    _recrear(cliente, coleccion(alias), secciones, dims)

    textos = [textos_cv(cv) for cv in cvs.values()]
    completos = {
        s: embeber(
            alias,
            [t[s] or SIN_INFO for t in textos],
            "documento",
            lote=1 if s == SECCION_COMPLETA else TAMANO_LOTE,  # el CV entero no cabe en lotes
        )
        for s in secciones
    }
    puntos = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, archivo)),
            vector={
                nombre_vector(s, d): recortar(completos[s][i], d)
                for s in secciones
                for d in dims
            },
            payload=payload_cv(archivo, cv),
        )
        for i, (archivo, cv) in enumerate(cvs.items())
    ]
    cliente.upsert(coleccion(alias), puntos)
    return len(puntos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embebe las hojas de vida y las guarda en Qdrant")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS, default=list(MODELOS))
    parser.add_argument(
        "--experimento",
        action="store_true",
        help="todos los tamaños y las secciones por separado además de 'completo' (más memoria)",
    )
    args = parser.parse_args()

    cvs = cargar("cvs")
    cliente = conectar()
    for alias in args.modelos:
        dims = MODELOS[alias].dims if args.experimento else None
        secciones = SECCIONES + (SECCION_COMPLETA,) if args.experimento else SECCIONES_PRODUCCION
        n = indexar(cliente, alias, cvs, dims, secciones)
        vectores = len(secciones) * len(dims or (1,))
        print(f"{alias}: {n} hojas de vida en '{coleccion(alias)}', {vectores} vectores por hoja")
    cliente.close()
