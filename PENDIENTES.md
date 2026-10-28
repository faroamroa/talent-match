# Pendientes

Estado al 2026-10-28. Fase actual: matching de hojas de vida y vacantes con Qdrant.

## Hecho

- `src/talentmatch/embeddings.py`: 3 modelos multilingües livianos con 3 tamaños de vector cada uno.

  | Alias    | Modelo                               | Parámetros       | Tamaños       | Matryoshka           |
  |----------|--------------------------------------|------------------|---------------|----------------------|
  | `gemma`  | `google/embeddinggemma-300m`         | 308M             | 768 / 512 / 256 | sí                   |
  | `e5`     | `intfloat/multilingual-e5-small`     | 118M             | 384 / 256 / 128 | no (control: recortar debería degradarlo) |
  | `potion` | `minishlab/potion-multilingual-128M` | 128M (estático)  | 256 / 128 / 64  | dimensiones ordenadas por PCA |

- `src/talentmatch/indexacion.py`: normalización (tildes, ciudades), campos derivados (`nivel_max`, `anios_exp`), textos por sección (`perfil`, `skills`, `educacion`) para hojas de vida y vacantes, payload e indexación en Qdrant local (`data/qdrant`). Una colección por modelo, con un vector por cada (sección, tamaño).
- `src/talentmatch/busqueda.py`: búsqueda con un `Prefetch` por sección y fusión RRF. Filtros `ninguno`, `estricto`, `tolerante` (el dato vacío pasa) y `suave` (bono en el puntaje, no descarta). Incluye el área metropolitana de Medellín y vacantes con varias ciudades.
- `src/talentmatch/evaluacion.py`: Recall@k y MRR por modelo × tamaño × filtro. Guarda un CSV en `salidas/`.
- `data/golden/golden_set.json`: 30 vacantes y 169 pares (hoja de vida, grado 1–2, razón), juzgados por Claude. Criterio: idoneidad del perfil; ciudad y modalidad no cuentan; los estudios incompletos sí.
- Comandos registrados en `pyproject.toml`: `talentmatch-indexar`, `talentmatch-buscar`, `talentmatch-evaluar`.

## Por hacer

1. **Resolver la falta de memoria al indexar (bloqueante).** Ningún modelo se ha indexado completo: las colecciones se crean pero quedan con 0 puntos. En WSL (7.8 GB) el kernel mató procesos por OOM (`memory.events: oom_kill 2`, pico de 7.87 GB) y tumbó la sesión, incluso con `potion`.
   - Ya se bajó el tamaño de lote a 4 (`TAMANO_LOTE` en `embeddings.py`), pero no se alcanzó a verificar.
   - Medir la memoria paso a paso con un límite para no tumbar la máquina:
     ```bash
     ulimit -v 4000000; OMP_NUM_THREADS=2 uv run talentmatch-indexar --modelos potion
     ```
   - Sospechas:
     - el texto de `perfil` es muy largo (incluye todas las descripciones de experiencia): truncarlo o embeber por experiencia;
     - los hilos de onnxruntime: probar `threads=2` en `TextEmbedding(...)`;
     - la carga simultánea de modelos (`cache` en `cargar`).
   - Alternativa: subir la memoria de WSL en `%UserProfile%\.wslconfig` (`[wsl2]` y `memory=12GB`), o correr en el otro computador.
2. **Indexar los 3 modelos y probar una búsqueda:**
   ```bash
   uv run talentmatch-indexar
   uv run talentmatch-buscar "senior data engineer" --modelo gemma --dim 256 --filtro tolerante
   ```
3. **Correr la evaluación y analizar la tabla:**
   ```bash
   uv run talentmatch-evaluar
   uv run talentmatch-evaluar --grado-min 2
   ```
   - ¿Cuánto recall pierde `estricto` frente a `tolerante` y `suave`?
   - ¿Cuánto cae `e5` al recortarlo (sin Matryoshka) frente a `gemma`?
4. **Revisar a mano 2 o 3 vacantes del golden set** para confirmar que el criterio de Claude coincide con el del reclutador.
5. **Evaluación más rápida:** guardar en caché los embeddings de las consultas. Hoy se re-embebe la misma vacante por cada tamaño y filtro.
6. **Bug en `pipeline.py`:** usa `logger.exception` pero `logger` no está definido (falta `import logging` y `logger = logging.getLogger(__name__)`).
7. Opcional:
   - vector disperso BM25 (`Qdrant/bm25`) para nombres exactos de herramientas;
   - normalizar los idiomas (`Inglés`/`English`/`Ingles`) para filtrar por idioma.

## Para trabajar desde otro computador

`data/` no se versiona, salvo `data/golden/`. Hay que copiar a mano `data/processed/extracciones/` (y `data/raw/` si se va a re-extraer). También hay que crear `.env` con `OPENROUTER_API_KEY` y `MODELO_LLM`. Después:

```bash
uv sync
```
