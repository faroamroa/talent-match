"""Relevantes del golden set que no quedan en el top 10: ¿falla el embedding o la extracción?

Para cada uno muestra su posición final (configuración de producción) y su posición en cada
sección por separado, y marca las secciones del CV con poco texto (señal de que la extracción
no capturó la información).

Requiere indexar con: uv run talentmatch-indexar --experimento
Uso: uv run python exploracion/analisis_errores.py [modelo] [dim]
"""

import sys
from collections import Counter

from talentmatch.busqueda import Ajustes, buscar
from talentmatch.embeddings import embeber_consulta, recortar
from talentmatch.evaluacion import relevantes_por_vacante
from talentmatch.indexacion import (
    SECCIONES,
    cargar,
    coleccion,
    conectar,
    nombre_vector,
    textos_cv,
    textos_vacante,
)

TOP = 10
MIN_CARACTERES = 40  # una sección más corta que esto casi no tiene información
TOTAL = 98


def posiciones_por_seccion(cliente, alias: str, dim: int, vacante: dict) -> dict[str, dict[str, int]]:
    """sección -> {archivo: posición} buscando solo con esa sección."""
    consultas = {s: t for s, t in textos_vacante(vacante).items() if s in SECCIONES and t}
    vectores = embeber_consulta(alias, tuple(consultas.values()))
    posiciones = {}
    for s, vector in zip(consultas, vectores):
        puntos = cliente.query_points(
            coleccion(alias), query=recortar(vector, dim), using=nombre_vector(s, dim), limit=TOTAL
        ).points
        posiciones[s] = {p.payload["archivo"]: i for i, p in enumerate(puntos, start=1)}
    return posiciones


def main() -> None:
    alias = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    vacantes = cargar("vacantes")
    cvs = cargar("cvs")
    relevancia = relevantes_por_vacante(grado_min=1)
    cliente = conectar()

    perdidos = 0
    total = 0
    mejor_seccion = Counter()  # en qué sección le iba mejor al relevante perdido
    secciones_pobres = Counter()
    for archivo_vac, relevantes in relevancia.items():
        vacante = vacantes[archivo_vac]
        ranking = buscar(cliente, vacante, alias, dim, TOTAL, "suave", Ajustes(candidatos=TOTAL))
        posicion = {r.archivo: i for i, r in enumerate(ranking, start=1)}
        por_seccion = posiciones_por_seccion(cliente, alias, dim, vacante)

        fuera = [cv for cv in relevantes if posicion.get(cv, TOTAL) > TOP]
        total += len(relevantes)
        perdidos += len(fuera)
        if not fuera:
            continue
        print(f"\n## {archivo_vac[:80]}  ({len(fuera)}/{len(relevantes)} fuera del top {TOP})")
        for cv in sorted(fuera, key=lambda c: posicion.get(c, TOTAL)):
            textos = textos_cv(cvs[cv])
            pobres = [s for s in SECCIONES if len(textos[s]) < MIN_CARACTERES]
            secciones_pobres.update(pobres)
            rangos = {s: por_seccion.get(s, {}).get(cv) for s in por_seccion}
            if rangos:
                mejor_seccion[min(rangos, key=lambda s: rangos[s] or TOTAL)] += 1
            detalle = "  ".join(f"{s}={r}" for s, r in rangos.items())
            aviso = f"  ⚠ poco texto en: {', '.join(pobres)}" if pobres else ""
            print(f"  {posicion.get(cv, '>'):>3}. {cv:<45} {detalle}{aviso}")
    cliente.close()

    print(f"\nResumen {alias} {dim}: {perdidos}/{total} relevantes fuera del top {TOP}")
    print(f"Sección donde mejor le iba a cada perdido: {dict(mejor_seccion)}")
    print(f"Secciones con poco texto entre los perdidos: {dict(secciones_pobres)}")


if __name__ == "__main__":
    main()
