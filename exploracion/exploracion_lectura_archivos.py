# Test 1
# Objetivo: ver qué formatos tienes y comprobar que pathlib
#  maneja tus nombres con tildes y emojis

from pathlib import Path
from collections import Counter

def raiz_proyecto() -> Path:
    for carpeta in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (carpeta / "pyproject.toml").exists():
            return carpeta
    raise FileNotFoundError("No se encontró pyproject.toml")

RAW = raiz_proyecto() / "data" / "raw"
print(RAW)

archivos = [p for p in RAW.rglob("*") if p.is_file()]
print(Counter(p.suffix.lower() for p in archivos))

# # # Test 2   - Se puede usar cuando haya un archivo con extension .txt para probar
# # Objetivo: leer texto sin romper las tildes, sin importar la codificación.

# # from charset_normalizer import from_path

# # resultado = from_path(raiz_proyecto() / "data" / "raw"/"vacantes"/"vacante1.txt").best()
# # print(resultado.encoding)   # ej. 'utf_8' o 'cp1252'
# # texto = str(resultado)
# # print(texto)

# # Test 3
# # Objetivo: ver si el texto sale limpio, vacío o mezclado.

# import pymupdf

# # doc = pymupdf.open(archivos[0])
# doc = pymupdf.open(RAW/"cvs"/"CV_Fernando_Roa_EN.pdf")
# for pagina in doc:
#     texto = pagina.get_text()
#     print(len(texto), "(cid:" in texto)
# print(doc[0].get_text()[:1000])


# # Test 4: página del PDF a imagen
# # Objetivo: generar lo que verá el LLM en las rutas híbrida y de imagen.

# SALIDA = raiz_proyecto() / "salidas" / "test4"
# SALIDA.mkdir(parents=True, exist_ok=True)   # crea la carpeta si no existe

# for dpi in (72, 150, 200):
#     pix = doc[0].get_pixmap(dpi=dpi)
#     png = pix.tobytes("png")

#     ruta = SALIDA / f"pagina1_{dpi}dpi.png"
#     ruta.write_bytes(png)

#     print(f"{dpi:>3} dpi | {pix.width}x{pix.height} px | {len(png)/1024:7.1f} KB | {ruta}")

# # # Test 5: DOCX
# # # Objetivo: comprobar que docx2python ve lo que python-docx no ve.
# # from docx2python import docx2python
# # import docx

# # p = next(a for a in archivos if a.suffix == ".docx")
# # with docx2python(p) as d:
# #     texto_1 = d.text
# # texto_2 = "\n".join(par.text for par in docx.Document(p).paragraphs)
# # print(len(texto_1), len(texto_2))

# # Test 6: imágenes
# # Objetivo: cargar la imagen y revisar su tamaño.
# # No hay texto que extraer, porque el LLM la leerá directamente.

# from io import BytesIO
# from PIL import Image

# EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}
# imagenes_raw = [p for p in archivos if p.suffix.lower() in EXT_IMAGEN]

# if imagenes_raw:
#     ruta_img = imagenes_raw[0]
# else:
#     ruta_img = SALIDA / "pagina1_150dpi.png"     # generada en el Test 4
#     print("No hay imágenes en data/raw; uso la del Test 4")

# # a) Desde archivo
# img_bytes = ruta_img.read_bytes()                 # bytes que se envían al LLM
# with Image.open(ruta_img) as img:
#     print(f"{ruta_img.name} | {img.size} | {img.mode} | {len(img_bytes)/1024:.1f} KB")

# # b) Directo desde memoria, con los bytes `png` del Test 4 (sin pasar por disco)
# with Image.open(BytesIO(png)) as img_mem:
#     print(f"Desde memoria | {img_mem.size} | {img_mem.mode} | {len(png)/1024:.1f} KB")
#     # png se crea en el bucle del test 4


# # Test 7: lector unificado
# # Objetivo: un solo punto de entrada para cualquier formato,
# #  sin importar si es hoja de vida o vacante.

# from pathlib import Path
# from collections import Counter

# import pymupdf
# from charset_normalizer import from_path
# from docx2python import docx2python
# import mammoth

# EXT_TEXTO = {".txt", ".md"}
# EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}


# def leer(ruta: Path) -> dict:
#     ext = ruta.suffix.lower()

#     if ext in EXT_TEXTO:
#         return {"texto": str(from_path(ruta).best()), "imagenes": []}

#     if ext == ".pdf":
#         with pymupdf.open(ruta) as doc:
#             return {
#                 "texto": "\n".join(p.get_text() for p in doc),
#                 "imagenes": [p.get_pixmap(dpi=150).tobytes("png") for p in doc],
#             }

#     if ext == ".docx":
#         try:
#             with docx2python(ruta) as d:
#                 texto = d.text
#         except Exception as e:
#             print(f"docx2python falló ({type(e).__name__}), uso mammoth: {ruta.name}")
#             with open(ruta, "rb") as f:
#                 texto = mammoth.extract_raw_text(f).value
#         return {"texto": texto, "imagenes": []}

#     if ext in EXT_IMAGEN:
#         return {"texto": "", "imagenes": [ruta.read_bytes()]}

#     raise ValueError(f"Formato no soportado: {ext}")


# resumen = Counter()
# for a in sorted(archivos):
#     try:
#         r = leer(a)
#     except Exception as e:
#         resumen["error"] += 1
#         print(f"ERROR | {a.name[:50]} | {e}")
#         continue

#     n_chars = len(r["texto"].strip())
#     cid = "(cid:" in r["texto"]
#     alerta = "SIN TEXTO" if n_chars < 200 else ("CID" if cid else "")
#     resumen[alerta or "ok"] += 1
#     print(f"{a.suffix:5} | {n_chars:6} chars | {len(r['imagenes'])} img | {alerta:9} | {a.name[:50]}")

# print(resumen)

# # Pruebas de la renderización de imágenes a partir de un pdf
# from talentmatch.lectura import renderizar_paginas

# pngs = renderizar_paginas(RAW / "cvs" / "CV_Fernando_Roa_EN.pdf")
# print(len(pngs), "páginas")
# print([round(len(p) / 1024) for p in pngs], "KB por página")

# SALIDA = raiz_proyecto() / "salidas" / "paso3"
# SALIDA.mkdir(parents=True, exist_ok=True)
# for i, png in enumerate(pngs, start=1):
#     (SALIDA / f"pagina{i}.png").write_bytes(png)


# # Prueba de extracción de información de una hoja de vida usando el LLM

# from talentmatch.lectura import preparar
# from talentmatch.extraccion import extraer
# from talentmatch.schemas import HojaDeVida, Vacante

# nombre_archivo = "CV_Fernando_Roa_EN.pdf"
# doc = preparar(RAW / "cvs" / nombre_archivo)
# print(f"Extrayendo información del archivo {RAW / "cvs" / nombre_archivo}")
# print(doc.modo, len(doc.texto), len(doc.imagenes))   # primero verifica la preparación, sin gastar
# datos, uso = extraer(doc, HojaDeVida)
# print(uso)
# print(datos.model_dump_json(indent=2))

# Prueba de extracción y guardado de información en archivos json

from datetime import datetime
from talentmatch.lectura import preparar
from talentmatch.extraccion import extraer
from talentmatch.almacenamiento import Metadatos, Extraccion, guardar
from talentmatch.schemas.hoja_vida import HojaDeVida
from talentmatch.schemas.vacante import Vacante
from talentmatch.config import CARPETA_EXTRACCIONES

ESQUEMAS = {"cvs": HojaDeVida, "vacantes": Vacante}

for tipo, esquema in ESQUEMAS.items():
    for archivo in sorted((RAW / tipo).iterdir())[:2]:      # [:2] → solo 2 por carpeta como prueba
        destino = CARPETA_EXTRACCIONES / tipo / f"{archivo.name}.json"
        if destino.exists():
            print(f"ya existe, salto: {archivo.name}")
            continue
        try:
            doc = preparar(archivo)
            datos, completion = extraer(doc, esquema)
            extraccion = Extraccion(
                metadatos=Metadatos(
                    archivo=doc.ruta.name,
                    tipo=doc.ruta.parent.name,
                    modo=doc.modo,
                    modelo=completion.model,
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                    fecha=datetime.now(),
                ),
                texto=doc.texto,
                datos=datos.model_dump(),
            )
            print("guardado:", guardar(extraccion = extraccion))
        except Exception:
            logger.exception("falló %s", archivo.name)