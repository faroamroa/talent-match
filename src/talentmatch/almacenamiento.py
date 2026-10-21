from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from talentmatch.lectura import Modo
from talentmatch.config import CARPETA_EXTRACCIONES

Tipo = Literal["cvs", "vacantes"]

class Metadatos(BaseModel):
    archivo: str
    tipo: Tipo
    modo: Modo
    modelo: str
    prompt_tokens: int
    completion_tokens: int
    fecha: datetime

class Extraccion(BaseModel):
    metadatos: Metadatos
    texto: str
    datos: dict


def guardar(extraccion:Extraccion, carpeta: Path = CARPETA_EXTRACCIONES) -> Path:
    destino = carpeta / extraccion.metadatos.tipo / f"{extraccion.metadatos.archivo}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(extraccion.model_dump_json(indent=2), encoding="utf-8")
    return destino