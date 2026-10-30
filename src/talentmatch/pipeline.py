import logging
from datetime import datetime

from talentmatch.almacenamiento import Extraccion, Metadatos, guardar
from talentmatch.config import CARPETA_EXTRACCIONES, RAW
from talentmatch.extraccion import extraer
from talentmatch.lectura import preparar
from talentmatch.schemas.hoja_vida import HojaDeVida
from talentmatch.schemas.vacante import Vacante

logger = logging.getLogger(__name__)

ESQUEMAS = {"cvs": HojaDeVida, "vacantes": Vacante}

def procesar(limite: int | None = None) -> None:

    for tipo, esquema in ESQUEMAS.items():
        for archivo in sorted((RAW / tipo).iterdir())[:limite]:
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

def main() -> None:
    procesar()