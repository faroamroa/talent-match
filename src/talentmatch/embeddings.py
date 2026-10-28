"""Modelos de embedding multilingües livianos y cómo prepararles el texto."""

from functools import cache
from typing import Literal

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from pydantic import BaseModel

Rol = Literal["consulta", "documento"]

# fastembed usa lotes de 256 por defecto: con textos largos agota la RAM (WSL con 8 GB se reinicia)
TAMANO_LOTE = 4


class ModeloEmbedding(BaseModel):
    """Modelo de fastembed con sus tamaños de vector y prefijos."""

    alias: str
    nombre: str
    dims: tuple[int, ...]  # el primero es el tamaño completo
    prefijo_consulta: str = ""
    prefijo_documento: str = ""
    personalizado: bool = False  # no viene en el catálogo de fastembed


MODELOS = {
    m.alias: m
    for m in [
        # 308M parámetros, entrenado con Matryoshka (768/512/256/128)
        ModeloEmbedding(
            alias="gemma",
            nombre="google/embeddinggemma-300m",
            dims=(768, 512, 256),
            prefijo_consulta="task: search result | query: ",
            prefijo_documento="title: none | text: ",
        ),
        # 118M parámetros, sin Matryoshka: recortar debería degradarlo
        ModeloEmbedding(
            alias="e5",
            nombre="intfloat/multilingual-e5-small",
            dims=(384, 256, 128),
            prefijo_consulta="query: ",
            prefijo_documento="passage: ",
            personalizado=True,
        ),
        # Modelo estático de 128M (sin transformer): dimensiones ordenadas por PCA
        ModeloEmbedding(
            alias="potion",
            nombre="minishlab/potion-multilingual-128M",
            dims=(256, 128, 64),
        ),
    ]
}


@cache
def cargar(alias: str) -> TextEmbedding:
    modelo = MODELOS[alias]
    if modelo.personalizado:
        TextEmbedding.add_custom_model(
            model=modelo.nombre,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=modelo.nombre),
            dim=modelo.dims[0],
            model_file="onnx/model.onnx",
        )
    return TextEmbedding(modelo.nombre)


def embeber(alias: str, textos: list[str], rol: Rol) -> list[list[float]]:
    """Embebe con el tamaño completo; los demás tamaños se obtienen con recortar()."""
    modelo = MODELOS[alias]
    prefijo = modelo.prefijo_consulta if rol == "consulta" else modelo.prefijo_documento
    textos = [prefijo + t for t in textos]
    return [v.tolist() for v in cargar(alias).embed(textos, batch_size=TAMANO_LOTE)]


def recortar(vector: list[float], dim: int) -> list[float]:
    """Matryoshka: los primeros valores resumen el significado. Qdrant (COSINE) renormaliza."""
    return vector[:dim]
