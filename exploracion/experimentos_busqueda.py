"""Compara variantes de la búsqueda contra el golden set (gemma 256 y potion 128).

Requiere indexar antes con: uv run talentmatch-indexar --experimento
Uso: uv run python exploracion/experimentos_busqueda.py
"""

from talentmatch.busqueda import Ajustes
from talentmatch.evaluacion import evaluar, guardar_csv, relevantes_por_vacante
from talentmatch.indexacion import SECCIONES, cargar, conectar

MODELOS = [("gemma", 256), ("potion", 128)]
BYTES_FLOAT = 4

# nombre -> (ajustes, filtros con los que tiene sentido probarlo)
EXPERIMENTOS = {
    "1 vector (completo, producción)": (Ajustes(), ["ninguno", "suave", "estricto", "tolerante"]),
    "3 vectores": (Ajustes(secciones=SECCIONES), ["ninguno", "suave"]),
    "2 vectores (sin educacion)": (Ajustes(secciones=("perfil", "skills")), ["ninguno", "suave"]),
    "perfil x2": (Ajustes(secciones=SECCIONES, pesos={"perfil": 2}), ["ninguno", "suave"]),
    "skills x2": (Ajustes(secciones=SECCIONES, pesos={"skills": 2}), ["ninguno", "suave"]),
    "educacion x0.5": (Ajustes(secciones=SECCIONES, pesos={"educacion": 0.5}), ["ninguno", "suave"]),
    "k_rrf 10": (Ajustes(secciones=SECCIONES, k_rrf=10), ["ninguno", "suave"]),
    "k_rrf 60": (Ajustes(secciones=SECCIONES, k_rrf=60), ["ninguno", "suave"]),
    "bono 0.02": (Ajustes(bono=0.02), ["suave"]),
    "bono 0.10": (Ajustes(bono=0.10), ["suave"]),
    "bono 0.20": (Ajustes(bono=0.20), ["suave"]),
}


def main() -> None:
    vacantes = cargar("vacantes")
    relevancia = {g: relevantes_por_vacante(g) for g in (1, 2)}
    cliente = conectar()
    filas = []
    for alias, dim in MODELOS:
        for nombre, (ajustes, filtros) in EXPERIMENTOS.items():
            for modo in filtros:
                fila = {
                    "modelo": alias,
                    "dim": dim,
                    "experimento": nombre,
                    "filtro": modo,
                    "KB por CV": round(len(ajustes.secciones) * dim * BYTES_FLOAT / 1024, 2),
                }
                for g, rel in relevancia.items():
                    m = evaluar(cliente, vacantes, rel, alias, dim, modo, ajustes)
                    fila |= {f"{metrica} g{g}": valor for metrica, valor in m.items()}
                filas.append(fila)
                print("  ".join(f"{c}={v}" for c, v in fila.items()), flush=True)
    cliente.close()
    print(f"\nGuardado: {guardar_csv(filas, 'experimentos')}")


if __name__ == "__main__":
    main()
