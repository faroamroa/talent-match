# src/talentmatch/schemas/comunes.py
"""Tipos compartidos entre los esquemas de hoja de vida y vacante."""

from enum import StrEnum

from pydantic import BaseModel, Field


# Enums
class Modalidad(StrEnum):
    """Lugar de trabajo. presencial: siempre en oficinas de la empresa;
    hibrido: algunos días en oficina y otros en casa; remoto: siempre desde casa."""

    PRESENCIAL = "presencial"
    HIBRIDO = "hibrido"
    REMOTO = "remoto"


class NivelEducativo(StrEnum):
    """Nivel de formación académica. pregrado = carrera universitaria profesional."""

    BACHILLER = "bachiller"
    TECNICO = "tecnico"
    TECNOLOGO = "tecnologo"
    PREGRADO = "pregrado"
    ESPECIALIZACION = "especializacion"
    MAESTRIA = "maestria"
    DOCTORADO = "doctorado"
    OTRO = "otro"


class NivelMCER(StrEnum):
    """Nivel de un idioma según el Marco Común Europeo de Referencia (MCER)."""

    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"
    NATIVO = "nativo"


class Moneda(StrEnum):
    """Moneda en código ISO 4217."""

    COP = "COP"
    USD = "USD"
    EUR = "EUR"
    OTRA = "otra"


class Periodicidad(StrEnum):
    """Periodo al que corresponde un monto de pago."""

    HORA = "hora"
    DIA = "dia"
    MES = "mes"
    ANIO = "anio"


# Modelos
class FechaParcial(BaseModel):
    """Fecha con precisión de mes. Si solo se conoce el año, mes es null.
    El mes en texto se pasa al número de mes equivalente"""

    anio: int | None = Field(None, ge=1900, le=2100, description="Año de 4 cifras")
    mes: int | None = Field(None, ge=1, le=12, description="Mes de 1 a 12")


class Ubicacion(BaseModel):
    """Ubicación geográfica."""

    pais: str | None = Field(None, description="País, ej. 'Colombia'")
    ciudad: str | None = Field(None, description="Ciudad, ej. 'Medellín'")


class Salario(BaseModel):
    """Monto o rango salarial. Si se da un único valor, usarlo en minimo y maximo."""

    minimo: float | None = Field(None, ge=0, description="Valor mínimo, solo el número")
    maximo: float | None = Field(None, ge=0, description="Valor máximo, solo el número")
    moneda: Moneda | None = None
    periodicidad: Periodicidad | None = None


class Idioma(BaseModel):
    """Idioma y nivel de dominio."""

    idioma: str | None = Field(None, description="Nombre del idioma, ej. 'Inglés'")
    nivel_texto: str | None = Field(
        None, description="Nivel tal como aparece en el documento, sin interpretar"
    )
    nivel_mcer: NivelMCER | None = Field(
        None,
        description="Nivel MCER solo si se menciona explícitamente. Elegir el más alto",
    )
