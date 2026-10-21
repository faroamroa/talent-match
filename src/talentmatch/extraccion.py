# Recibe un Documento y un esquema (HojaDeVida o Vacante) y devuelve el esquema lleno
# Instructor se encarga de tres cosas que haría a mano:
# 1. Convierte tu modelo de Pydantic en un JSON schema y se lo pasa al LLM.
# 2. Valida la respuesta contra el modelo.
# 3. Si la validación falla, reintenta y le muestra al LLM el error para que lo corrija.

# Documento ─► elegir_ruta() ─► (renderizar si hace falta) ─► armar mensaje ─► instructor + LLM ─► HojaDeVida/Vacante

import base64
import os

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from talentmatch.lectura import Documento, elegir_modo, renderizar_paginas

load_dotenv()

MODELO = os.environ["MODELO_LLM"]
cliente = instructor.from_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
    )
    )

# Instrucciones del sistema
INSTRUCCIONES = (
    "Eres un extractor de información de documentos de reclutamiento. "
    "Extrae solo datos que aparezcan explícitamente en el documento. "
    "Si un dato no aparece, devuelve null o una lista vacía. No inventes ni infieras. "
    "Conserva nombres propios, correos y teléfonos exactamente como aparecen."
)

# Armar el contenido del mensaje
# Ejemplo:
# [
#     {"type": "text", "text": "…texto del CV…"},
#     {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG…"}},
#     {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}},
# ]

def imagen_a_data_url(imagen: bytes, mime:str = "image/png") -> str:
    b64 = base64.b64encode(imagen).decode()
    return f"data:{mime};base64,{b64}"

def armar_contenido(doc: Documento) -> list[dict]:
    partes = []
    # Si el documento tiene texto
    if elegir_modo(doc) in ("texto", "hibrida"):
        partes.append({"type":"text", "text":doc.texto})
    # Si el documento tiene imagenes
    if elegir_modo(doc) in ("imagen", "hibrida"):
        for imagen in doc.imagenes:
            partes.append({"type":"image_url", "image_url":{"url":imagen_a_data_url(imagen=imagen)}})
    return partes

def extraer(doc: Documento, esquema: type[BaseModel]):
    mensajes = [
        {"role":"system", "content": INSTRUCCIONES},
        {"role":"user", "content":armar_contenido(doc)}
    ]
    datos, completion = cliente.chat.completions.create_with_completion(
        model=MODELO,
        response_model=esquema,
        messages=mensajes,
        max_retries=2,
        temperature=0
    )
    print(completion.model)                        # qué modelo respondió realmente
    print(completion.usage)                        # tokens enviados y recibidos
    print(completion.choices[0].finish_reason)     # por qué terminó
    print(completion.choices[0].message)           # lo que respondió, en crudo
    return datos, completion.usage