# src/talentmatch/schemas/vacante.py
"""Esquema de extracción de una vacante."""

from enum import StrEnum

from pydantic import BaseModel, Field

from .comunes import Idioma, Modalidad, NivelEducativo, Salario, Ubicacion


# Enums propios
class Seniority(StrEnum):
    """Nivel de experiencia del cargo. lider: coordina un equipo;
    directivo: gerente, director o superior."""

    PRACTICANTE = "practicante"
    JUNIOR = "junior"
    SEMI_SENIOR = "semi_senior"
    SENIOR = "senior"
    LIDER = "lider"
    DIRECTIVO = "directivo"
    OTRO = "otro"


class TipoContrato(StrEnum):
    """Tipo de vinculación laboral."""

    INDEFINIDO = "indefinido"
    TERMINO_FIJO = "termino_fijo"
    OBRA_LABOR = "obra_labor"
    PRESTACION_SERVICIOS = "prestacion_servicios"
    APRENDIZAJE = "aprendizaje"
    PRACTICAS = "practicas"
    OTRO = "otro"


# Submodelos
class CondicionesContrato(BaseModel):
    """Condiciones laborales ofrecidas."""

    modalidad: Modalidad | None = None
    tipo_contrato: TipoContrato | None = None
    duracion: str | None = Field(
        None, description="Duración tal como aparece, ej. '6 meses'; null si es indefinido"
    )
    jornada: str | None = Field(None, description="Horario o jornada, ej. 'tiempo completo'")
    salario: Salario | None = None


class ExperienciaEspecifica(BaseModel):
    """Experiencia exigida en un cargo o área concreta."""

    area_o_cargo: str | None = Field(None, description="Ej. 'desarrollo backend', 'analista de datos'")
    anios_minimos: float | None = Field(None, ge=0, description="Años mínimos; 6 meses = 0.5")


# Modelo raíz
class Vacante(BaseModel):
    """Información de una oferta de empleo. Extrae solo lo que aparece explícitamente
    en el documento; si un dato no está, déjalo en null o en lista vacía. Ignora menús,
    publicidad y otras vacantes sugeridas por el portal."""

    titulo: str | None = Field(None, description="Título del cargo")
    empresa: str | None = Field(None, description="Empresa que contrata, no el portal")
    portal_origen: str | None = Field(
        None, description="Portal donde se publicó, ej. 'LinkedIn', 'Magneto', 'Comfama'"
    )
    codigo_vacante: str | None = Field(None, description="Identificador de la vacante en el portal")
    ubicacion: Ubicacion | None = None
    descripcion_cargo: str | None = Field(None, description="Resumen del propósito del cargo")
    responsabilidades: list[str] = Field(default_factory=list)
    seniority: Seniority | None = None
    condiciones: CondicionesContrato | None = None
    beneficios: list[str] = Field(default_factory=list)

    # Requisitos
    nivel_educativo_minimo: NivelEducativo | None = None
    profesiones_requeridas: list[str] = Field(
        default_factory=list, description="Carreras o títulos aceptados, ej. 'Ingeniería de Sistemas'"
    )
    anios_experiencia_general: float | None = Field(
        None, ge=0, description="Años mínimos de experiencia laboral total"
    )
    experiencia_especifica: list[ExperienciaEspecifica] = Field(default_factory=list)
    idiomas_obligatorios: list[Idioma] = Field(default_factory=list)
    idiomas_deseables: list[Idioma] = Field(default_factory=list)
    herramientas_obligatorias: list[str] = Field(
        default_factory=list, description="Software, lenguajes y tecnologías exigidos"
    )
    herramientas_deseables: list[str] = Field(
        default_factory=list, description="Software, lenguajes y tecnologías deseables o 'plus'"
    )
    habilidades_obligatorias: list[str] = Field(default_factory=list)
    habilidades_deseables: list[str] = Field(default_factory=list)
    certificaciones_requeridas: list[str] = Field(default_factory=list)
