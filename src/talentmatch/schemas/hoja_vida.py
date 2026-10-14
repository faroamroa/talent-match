# src/talentmatch/schemas/hoja_vida.py
"""Esquema de extracción de una hoja de vida."""

from enum import StrEnum

from pydantic import BaseModel, Field

from .comunes import (
    FechaParcial,
    Idioma,
    Modalidad,
    NivelEducativo,
    Salario,
    Ubicacion,
)


# Enums propios
class TipoDocumento(StrEnum):
    """Tipo de documento de identidad. cc: cédula de ciudadanía; ce: cédula de
    extranjería; ti: tarjeta de identidad; ppt: permiso por protección temporal;
    dui: documento único de identidad (El Salvador)."""

    CC = "cc"
    CE = "ce"
    TI = "ti"
    PASAPORTE = "pasaporte"
    PPT = "ppt"
    DUI = "dui"
    OTRO = "otro"


class EstadoEstudio(StrEnum):
    """finalizado: culminado; en_curso: se cursa actualmente;
    incompleto: no se cursa actualmente y no fue culminado."""

    FINALIZADO = "finalizado"
    EN_CURSO = "en_curso"
    INCOMPLETO = "incompleto"


class TipoEnlace(StrEnum):
    """Plataforma a la que pertenece un enlace."""

    LINKEDIN = "linkedin"
    GITHUB = "github"
    PORTAFOLIO = "portafolio"
    OTRO = "otro"


# Submodelos
class DocumentoIdentidad(BaseModel):
    """Documento de identidad de la persona."""

    tipo: TipoDocumento | None = None
    numero: str | None = Field(None, description="Número tal como aparece, como texto")


class Enlace(BaseModel):
    """Enlace a un perfil profesional o evidencia de trabajo."""

    tipo: TipoEnlace | None = None
    url: str | None = Field(None, description="URL completa tal como aparece")


class Contacto(BaseModel):
    """Datos de contacto."""

    telefonos: list[str] = Field(default_factory=list, description="Teléfonos o celulares")
    correos: list[str] = Field(default_factory=list, description="Correos electrónicos")
    enlaces: list[Enlace] = Field(default_factory=list)


class Estudio(BaseModel):
    """Formación académica formal (no cursos cortos ni certificaciones)."""

    titulo: str | None = Field(None, description="Título o programa, ej. 'Ingeniería de Sistemas'")
    institucion: str | None = None
    nivel: NivelEducativo | None = None
    estado: EstadoEstudio | None = None
    fecha_inicio: FechaParcial | None = None
    fecha_fin: FechaParcial | None = Field(None, description="null si está en curso")


class Experiencia(BaseModel):
    """Experiencia laboral en un cargo."""

    cargo: str | None = None
    empresa: str | None = None
    ubicacion: Ubicacion | None = None
    fecha_inicio: FechaParcial | None = None
    fecha_fin: FechaParcial | None = Field(None, description="null si es el trabajo actual")
    es_actual: bool | None = Field(
        None, description="true si el documento indica que sigue en el cargo (ej. 'actual')"
    )
    descripcion: str | None = Field(None, description="Funciones y logros del cargo")
    herramientas: list[str] = Field(
        default_factory=list, description="Software o tecnologías usadas en este cargo"
    )


class Certificacion(BaseModel):
    """Certificación o curso complementario."""

    nombre: str | None = None
    entidad: str | None = Field(None, description="Entidad que la emite")
    anio: int | None = Field(None, ge=1900, le=2100)


# Modelo raíz
class HojaDeVida(BaseModel):
    """Información de una hoja de vida. Extrae solo lo que aparece explícitamente
    en el documento; si un dato no está, déjalo en null o en lista vacía."""

    nombre_completo: str | None = None
    documento_identidad: DocumentoIdentidad | None = None
    nacionalidad: str | None = None
    contacto: Contacto | None = None
    residencia: Ubicacion | None = Field(None, description="Lugar donde vive la persona")
    titular: str | None = Field(None, description="Cargo actual o titular del perfil")
    resumen_perfil: str | None = None
    educacion: list[Estudio] = Field(default_factory=list)
    experiencia: list[Experiencia] = Field(default_factory=list)
    idiomas: list[Idioma] = Field(default_factory=list)
    certificaciones: list[Certificacion] = Field(default_factory=list)
    herramientas: list[str] = Field(
        default_factory=list, description="Software, lenguajes y tecnologías que maneja"
    )
    habilidades: list[str] = Field(
        default_factory=list, description="Habilidades técnicas y blandas, ej. 'liderazgo'"
    )
    preferencia_modalidad: Modalidad | None = None
    aspiracion_salarial: Salario | None = None
    idioma_documento: str | None = Field(
        None, description="Idioma en que está escrito el documento, código ISO 639-1 (es, en...)"
    )
