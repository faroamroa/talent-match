# Lectura de documentos en cualquier formato
# Objetivo: un solo punto de entrada para cualquier formato,
#  sin importar si es hoja de vida o vacante.

import logging
from pathlib import Path
from typing import Literal

import mammoth
import pymupdf
from charset_normalizer import from_path
from docx2python import docx2python
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EXT_TEXTO = {".txt", ".md"}
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}
Modo = Literal["texto", "imagen", "hibrida"]
MIN_CARACTERES: int = 200
DPI: int = 150
MAX_PAGINAS: int = 5


class Documento(BaseModel):
    ruta: Path
    texto: str
    imagenes: list[bytes] = Field(
        default_factory=list,
        description="Lista de imágenes en bytes asociadas al documento",
    )
    modo: Modo | None = None


def leer(ruta: Path) -> Documento:
    ext = ruta.suffix.lower()

    if ext in EXT_TEXTO:
        texto = str(from_path(ruta).best())
        return Documento(
            ruta=ruta,
            texto=texto,
        )

    if ext == ".pdf":
        with pymupdf.open(ruta) as doc:
            texto = "\n".join(p.get_text() for p in doc)
            return Documento(
                ruta=ruta,
                texto=texto,
            )

    if ext == ".docx":
        try:
            with docx2python(ruta) as d:
                texto = d.text
        except Exception:
            logger.warning(
                "docx2python falló en %s, uso mammoth", ruta.name, exc_info=True
            )
            with open(ruta, "rb") as f:
                texto = mammoth.extract_raw_text(f).value
        return Documento(
            ruta=ruta,
            texto=texto,
        )

    if ext in EXT_IMAGEN:
        texto = ""
        imagenes = [ruta.read_bytes()]
        return Documento(
            ruta=ruta, 
            texto=texto, 
            imagenes=imagenes)

    raise ValueError(f"Formato no soportado: {ext}")


# Definir la ruta o input dependiendo del tipo de documento
# Se decide entre "texto", "hibrida" o "imagen"


def elegir_modo(doc: Documento) -> Modo:
    ext = doc.ruta.suffix.lower()

    if ext in EXT_IMAGEN:
        return "imagen"
    if ext in EXT_TEXTO or ext == ".docx":
        return "texto"
    texto_inutil = len(doc.texto.strip()) < MIN_CARACTERES or "(cid:" in doc.texto
    if texto_inutil:
        return "imagen"

    return "hibrida"


# Renderizar páginas. De acuerdo con el documento cuando el retorno se
# indique como "hibrida" o "imagen" se deben renderizar imágenes


def renderizar_paginas(
    ruta:Path, dpi: int = DPI, max_paginas: int = MAX_PAGINAS
) -> list[bytes]:
    imagenes: list = []
    with pymupdf.open(ruta) as doc:
        for pagina in doc.pages(0, max_paginas):
            png = pagina.get_pixmap(dpi=dpi).tobytes("png")
            imagenes.append(png)
        # imagenes = [pagina.get_pixmap(dpi=dpi).tobytes("png") for pagina in doc.pages(0, max_paginas)]
        return imagenes

def preparar(ruta:Path)->Documento:
    doc = leer(ruta)
    doc.modo = elegir_modo(doc)
    if doc.modo != "texto" and ruta.suffix.lower()==".pdf":
        doc.imagenes=renderizar_paginas(doc.ruta)
    return doc