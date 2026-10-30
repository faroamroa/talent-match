"""Recall@k y MRR contra el golden set, comparando modelos, tamaños de vector y filtros."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from qdrant_client import QdrantClient

from talentmatch.busqueda import MODOS, Ajustes, ModoFiltro, buscar
from talentmatch.config import RAIZ
from talentmatch.embeddings import MODELOS
from talentmatch.indexacion import cargar, conectar

GOLDEN = RAIZ / "data" / "golden" / "golden_set.json"
SALIDAS = RAIZ / "salidas"


def recall_at_k(ranking: list[str], relevantes: set[str], k: int) -> float:
    return len(set(ranking[:k]) & relevantes) / len(relevantes)


def reciprocal_rank(ranking: list[str], relevantes: set[str]) -> float:
    for posicion, archivo in enumerate(ranking, start=1):
        if archivo in relevantes:
            return 1 / posicion
    return 0.0


def relevantes_por_vacante(grado_min: int, ruta: Path = GOLDEN) -> dict[str, set[str]]:
    """Solo vacantes con al menos un relevante: con cero, recall dividiría por cero."""
    golden = json.loads(ruta.read_text(encoding="utf-8"))
    por_vacante = {
        vacante: {cv for cv, j in datos["relevantes"].items() if j["grado"] >= grado_min}
        for vacante, datos in golden["vacantes"].items()
    }
    return {v: r for v, r in por_vacante.items() if r}


def evaluar(
    cliente: QdrantClient,
    vacantes: dict[str, dict],
    relevancia: dict[str, set[str]],
    alias: str,
    dim: int,
    modo: ModoFiltro,
    ajustes: Ajustes | None = None,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Promedio de Recall@k y MRR sobre las vacantes del golden set."""
    recalls = {k: [] for k in ks}
    rrs = []
    for archivo, relevantes in relevancia.items():
        resultados = buscar(cliente, vacantes[archivo], alias, dim, max(ks), modo, ajustes)
        ranking = [r.archivo for r in resultados]
        for k in ks:
            recalls[k].append(recall_at_k(ranking, relevantes, k))
        rrs.append(reciprocal_rank(ranking, relevantes))
    metricas = {f"R@{k}": round(mean(v), 3) for k, v in recalls.items()}
    metricas["MRR"] = round(mean(rrs), 3)
    return metricas


def guardar_csv(filas: list[dict], prefijo: str) -> Path:
    SALIDAS.mkdir(exist_ok=True)
    destino = SALIDAS / f"{prefijo}_{datetime.now():%Y%m%d_%H%M}.csv"
    with destino.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=filas[0])
        escritor.writeheader()
        escritor.writerows(filas)
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa la búsqueda contra el golden set")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS, default=list(MODELOS))
    parser.add_argument("--filtros", nargs="+", choices=MODOS, default=list(MODOS))
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--grado-min", type=int, choices=[1, 2], default=1)
    parser.add_argument("--golden", type=Path, default=GOLDEN, help="p. ej. data/golden/golden_set_v1.json")
    parser.add_argument(
        "--todos-los-tamanos",
        action="store_true",
        help="evalúa todos los tamaños (requiere indexar con --experimento)",
    )
    args = parser.parse_args()

    relevancia = relevantes_por_vacante(args.grado_min, args.golden)
    vacantes = cargar("vacantes")
    print(f"{len(relevancia)} vacantes con relevantes de grado >= {args.grado_min}\n")

    cliente = conectar()
    filas = []
    for alias in args.modelos:
        dims = MODELOS[alias].dims if args.todos_los_tamanos else (MODELOS[alias].dim_produccion,)
        for dim in dims:
            for modo in args.filtros:
                fila = {"modelo": alias, "dim": dim, "filtro": modo}
                fila |= evaluar(cliente, vacantes, relevancia, alias, dim, modo, ks=tuple(args.ks))
                filas.append(fila)
                print("  ".join(f"{c}={v}" for c, v in fila.items()))
    cliente.close()
    print(f"\nGuardado: {guardar_csv(filas, 'evaluacion')}")
