# Pendientes

Estado al 2026-10-29. Fase actual: matching de hojas de vida y vacantes con Qdrant.

## Hecho

- `src/talentmatch/embeddings.py`: 3 modelos multilingües livianos. Cada uno tiene un tamaño de producción (`dim_produccion`), que es el único que se indexa por defecto, y otros tamaños para experimentar. `embeber_consulta()` guarda en caché las últimas 256 consultas.

  | Alias    | Modelo                               | Parámetros      | Tamaños         | Producción | Matryoshka |
  |----------|--------------------------------------|-----------------|-----------------|------------|------------|
  | `gemma`  | `google/embeddinggemma-300m`         | 308M            | 768 / 512 / 256 | 256 | sí |
  | `e5`     | `intfloat/multilingual-e5-small`     | 118M            | 384 / 256 / 128 | 384 | no |
  | `potion` | `minishlab/potion-multilingual-128M` | 128M (estático) | 256 / 128 / 64  | 128 | dimensiones ordenadas por PCA |

- `src/talentmatch/indexacion.py`:
  - Normalización de tildes, ciudades e idiomas (`Inglés`/`English`/`Ingles` → `ingles`; el MCER también se lee del texto).
  - Campos derivados: `nivel_max` y `anios_exp`.
  - **Producción: 1 vector `completo`** (todo el CV) por hoja de vida, decidido el 2026-10-29. Las secciones `perfil`, `skills` y `educacion` por separado solo se indexan con `--experimento`.
- `src/talentmatch/busqueda.py`:
  - RRF sobre las secciones, con filtros `ninguno`, `estricto`, `tolerante` y `suave`.
  - `Ajustes`: secciones, pesos y `k` de RRF, y bono.
- `src/talentmatch/evaluacion.py`: Recall@k y MRR, con `--golden` y `--todos-los-tamanos`.
- `exploracion/experimentos_busqueda.py`: compara variantes y reporta los KB por CV.
- `exploracion/analisis_errores.py`: posición por sección de los relevantes perdidos.
- **Golden set** (`data/golden/`):
  - `golden_set_v1.json`: 169 pares, juzgados por Claude. Una auditoría encontró sesgo hacia las palabras clave.
  - `golden_set_v2.json`: 293 pares, re-juzgados por 3 agentes Opus. Demasiado laxo: 9.8 por vacante.
  - `golden_set.json` (**v3, el activo**): máximo 3 por vacante, los de encaje más directo. Son 89 pares (51 de grado 2 y 38 de grado 1).
- **Descartado para no agregar complejidad sin beneficio:**
  - **`perfil_corto`** (titular + cargos). Subía el MRR de gemma de 0.833 a 0.912 con la v2, pero sumaba un vector más y la mejora podía ser sesgo de selección sobre las mismas 30 vacantes. Vale la pena re-probarlo cuando exista el híbrido con BM25.
  - **Experiencia afín por cargo** (colección `exp_*`, unos 4.5 vectores por CV). No mejoró ninguna métrica.
- **Memoria al indexar:** se usan lotes de 4 y lote de 1 para el texto `completo`. Con lotes más grandes, gemma superó el tope de 3.5 GB. En modo `--experimento`, gemma tarda 4.3 minutos con un pico de 1.6 GB.

## Resultados con el golden set v3

Tablas completas en `salidas/eval_v3_g*.log` y `salidas/experimentos_v3.log`. Grado ≥ 1, 30 vacantes:

| Configuración (gemma 256) | KB por CV | R@5 | R@10 | MRR | R@10 grado 2 | MRR grado 2 |
|---|---|---|---|---|---|---|
| 3 vectores, sin filtro | 3.0 | 0.383 | **0.617** | 0.566 | **0.722** | 0.458 |
| **1 vector `completo`, sin filtro** | **1.0** | **0.439** | 0.594 | **0.659** | 0.632 | **0.510** |
| 2 vectores (sin educación) | 2.0 | 0.461 | 0.606 | 0.535 | 0.639 | 0.415 |
| 3 vectores, estricto | 3.0 | 0.278 | 0.400 | 0.394 | 0.479 | 0.387 |

- **1 vector frente a 3 vectores:** con un tercio de la memoria, `completo` pone antes al primer relevante (MRR +0.09) y tiene mejor R@5. Pierde algo de R@10, sobre todo con los grado 2 (0.632 frente a 0.722). Con potion pasa lo mismo.
- **gemma sigue ganando** con claridad, y 256 rinde igual que 768 (MRR 0.566 frente a 0.543).
- **El filtro estricto sigue siendo siempre el peor:** pierde más de 0.2 de R@10.
- **Pesos por sección, `k_rrf 10` y bono:** cambian los resultados en ±0.03, es decir, ruido. `k_rrf 60` empeora todo.
- **Análisis de errores (gemma 256, 3 vectores):** 33 de 89 relevantes quedan fuera del top 10. Fallan por igual en `perfil` y `skills`, y solo 2 tienen textos pobres. La extracción no es el cuello de botella.

## Por hacer

1. **Cuantización:** int8 (4 veces menos memoria) o binaria (32 veces) con rescoring. El modo local de Qdrant no la aplica, así que para medirla hay que levantar Qdrant en Docker.
2. **Híbrido con BM25** (`Qdrant/bm25`, vector disperso), para nombres exactos de herramientas. Después, re-probar `perfil_corto`.
3. **Revisar a mano el golden set v3**, sobre todo las 6 vacantes sin grado 2: SAP, Seguridad TI, Director comercial, Director de proyectos, Convenios y Seguros.
4. Opcional: filtro por idioma obligatorio, que ya es posible con el payload `idiomas: [{idioma, mcer}]`.

## Cómo correrlo

```bash
uv run talentmatch-indexar                      # producción: 1 tamaño por modelo
uv run talentmatch-indexar --experimento        # todos los tamaños + sección 'completo'
uv run talentmatch-buscar "senior data engineer" --modelo gemma --filtro suave
uv run talentmatch-buscar "senior data engineer" --secciones completo
uv run talentmatch-evaluar --modelos gemma --grado-min 2
uv run talentmatch-evaluar --todos-los-tamanos --golden data/golden/golden_set_v2.json
uv run python exploracion/experimentos_busqueda.py      # requiere --experimento
uv run python exploracion/analisis_errores.py gemma 256
```

Para que un exceso de memoria no tumbe la sesión, se puede limitar así:

```bash
systemd-run --user --scope -p MemoryMax=3500M uv run talentmatch-indexar --experimento
```

## Para trabajar desde otro computador

`data/` no se versiona, salvo `data/golden/`. Hay que copiar `data/raw/` y `data/processed/extracciones/`. Al re-extraer, las extracciones cambian y el golden set deja de ser comparable. `data/qdrant/` no se copia: se regenera. Después:

```bash
uv sync
uv run talentmatch-indexar
```
