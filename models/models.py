from pydantic import BaseModel
from typing import Optional


class Metadata(BaseModel):
    titulo: str
    area: str
    tipo_de_texto: str
    chunk: int


class Record(BaseModel):
    id: str
    text: str
    metadata: Metadata


class TextRecord(BaseModel):
    titulo: str
    area: str
    tipo_de_texto: str
    contenido: Optional[str] = None
