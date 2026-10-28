"""Búsqueda de hojas de vida para una vacante: vectores por sección, fusión RRF y filtros."""

import argparse
from typing import Literal

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from talentmatch.embeddings import MODELOS, embeber, recortar
from talentmatch.indexacion import (
    ORDEN_NIVEL,
    SECCIONES,
    cargar,
    ciudades,
    coleccion,
    conectar,
    nombre_vector,
    normalizar,
    textos_vacante,
)

ModoFiltro = Literal["ninguno", "estricto", "tolerante", "suave"]
MODOS: tuple[ModoFiltro, ...] = ("ninguno", "estricto", "tolerante", "suave")

# El primer puesto de una lista RRF suma 0.5, el décimo ~0.09: el bono reordena sin descartar
BONO = 0.05

AREA_METROPOLITANA = {
    "medellin": [
        "medellin", "envigado", "bello", "itagui", "sabaneta",
        "la estrella", "caldas", "copacabana", "girardota", "barbosa",
    ],
}


class Requisitos(BaseModel):
    """Restricciones duras de la vacante, ya normalizadas. None = la vacante no lo exige."""

    ciudades: list[str] | None = None
    nivel_min: int | None = None
    anios_min: float | None = None


class Resultado(BaseModel):
    archivo: str
    puntaje: float
    payload: dict


def requisitos(v: dict) -> Requisitos:
    lista = ciudades((v["ubicacion"] or {}).get("ciudad"))
    ampliadas = sorted({c for ciudad in lista for c in AREA_METROPOLITANA.get(ciudad, [ciudad])})
    return Requisitos(
        ciudades=ampliadas or None,
        nivel_min=ORDEN_NIVEL.get(v["nivel_educativo_minimo"]),
        anios_min=v["anios_experiencia_general"],
    )


def _condiciones(req: Requisitos) -> list[tuple[str, models.FieldCondition]]:
    """(campo del payload, condición) solo para lo que la vacante sí exige."""
    pares = []
    if req.ciudades:
        pares.append(("ciudad", models.FieldCondition(key="ciudad", match=models.MatchAny(any=req.ciudades))))
    if req.nivel_min:
        pares.append(("nivel_max", models.FieldCondition(key="nivel_max", range=models.Range(gte=req.nivel_min))))
    if req.anios_min:
        pares.append(("anios_exp", models.FieldCondition(key="anios_exp", range=models.Range(gte=req.anios_min))))
    return pares


def filtro_para(req: Requisitos, modo: ModoFiltro) -> models.Filter | None:
    """estricto: debe cumplir. tolerante: debe cumplir o no tener el dato. ninguno/suave: sin filtro."""
    pares = _condiciones(req)
    if modo in ("ninguno", "suave") or not pares:
        return None
    if modo == "estricto":
        return models.Filter(must=[c for _, c in pares])
    return models.Filter(
        must=[
            models.Filter(should=[c, models.IsEmptyCondition(is_empty=models.PayloadField(key=campo))])
            for campo, c in pares
        ]
    )


def _bono(payload: dict, req: Requisitos) -> float:
    """Cumple: BONO completo. Dato desconocido: medio BONO. No cumple: 0."""
    total = 0.0
    chequeos = [
        ("ciudad", req.ciudades, lambda x: x in req.ciudades),
        ("nivel_max", req.nivel_min, lambda x: x >= req.nivel_min),
        ("anios_exp", req.anios_min, lambda x: x >= req.anios_min),
    ]
    for campo, exigido, cumple in chequeos:
        if not exigido:
            continue
        valor = payload.get(campo)
        total += BONO / 2 if valor is None else BONO * cumple(valor)
    return total


def buscar(
    cliente: QdrantClient,
    vacante: dict,
    alias: str,
    dim: int | None = None,
    k: int = 10,
    modo: ModoFiltro = "ninguno",
    secciones: tuple[str, ...] = SECCIONES,
    candidatos: int = 50,
) -> list[Resultado]:
    """Una búsqueda por sección de la vacante (Prefetch) y fusión de rankings con RRF."""
    dim = dim or MODELOS[alias].dims[0]
    req = requisitos(vacante)
    filtro = filtro_para(req, modo)

    consultas = {s: t for s, t in textos_vacante(vacante).items() if s in secciones and t}
    vectores = embeber(alias, list(consultas.values()), "consulta")
    prefetch = [
        models.Prefetch(
            query=recortar(vector, dim),
            using=nombre_vector(s, dim),
            filter=filtro,
            limit=candidatos,
        )
        for s, vector in zip(consultas, vectores)
    ]
    respuesta = cliente.query_points(
        coleccion(alias),
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidatos if modo == "suave" else k,
    )
    resultados = [
        Resultado(archivo=p.payload["archivo"], puntaje=p.score, payload=p.payload)
        for p in respuesta.points
    ]
    if modo == "suave":
        for r in resultados:
            r.puntaje += _bono(r.payload, req)
        resultados.sort(key=lambda r: r.puntaje, reverse=True)
    return resultados[:k]


def encontrar_vacante(texto: str, vacantes: dict[str, dict]) -> str:
    """Busca por fragmento del nombre de archivo, sin importar tildes ni mayúsculas."""
    clave = normalizar(texto) or ""
    coincidencias = [a for a in vacantes if clave in (normalizar(a.replace(",", " ")) or "")]
    if len(coincidencias) != 1:
        opciones = "\n  ".join(coincidencias or vacantes)
        raise SystemExit(f"'{texto}' coincide con {len(coincidencias)} vacantes:\n  {opciones}")
    return coincidencias[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca las mejores hojas de vida para una vacante")
    parser.add_argument("vacante", help="fragmento del nombre de archivo, ej. 'senior data engineer'")
    parser.add_argument("--modelo", choices=MODELOS, default="gemma")
    parser.add_argument("--dim", type=int, help="tamaño de vector; por defecto el completo")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--filtro", choices=MODOS, default="tolerante")
    parser.add_argument("--secciones", nargs="+", choices=SECCIONES, default=list(SECCIONES))
    args = parser.parse_args()

    if args.dim and args.dim not in MODELOS[args.modelo].dims:
        parser.error(f"--dim para {args.modelo} debe ser uno de {MODELOS[args.modelo].dims}")

    vacantes = cargar("vacantes")
    archivo = encontrar_vacante(args.vacante, vacantes)
    vacante = vacantes[archivo]
    print(f"Vacante: {archivo}\nRequisitos: {requisitos(vacante).model_dump()}\n")

    cliente = conectar()
    resultados = buscar(
        cliente, vacante, args.modelo, args.dim, args.k, args.filtro, tuple(args.secciones)
    )
    cliente.close()
    for i, r in enumerate(resultados, start=1):
        p = r.payload
        print(
            f"{i:>2}. {r.puntaje:.3f}  {r.archivo:<45} | {p['ciudad'] or '¿?':<12} "
            f"| nivel {p['nivel_max'] or '¿?'} | {p['anios_exp'] or '¿?'} años | {p['titular'] or ''}"
        )
