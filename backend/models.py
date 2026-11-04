"""Validated public JSON contracts; musical objects live in the team's Schema.py."""
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from Schema import Melody, Note


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class NoteInput(Model):
    midi: int = Field(ge=0, le=127, strict=True)
    duration: float = Field(gt=0, le=32)


class MelodyInput(Model):
    title: str = Field(default='Untitled', min_length=1, max_length=120)
    tonic: int = Field(default=0, ge=0, le=11, strict=True)
    mode: Literal['major', 'minor'] = 'major'
    time_signature: tuple[int, int] = (4, 4)
    harmonic_rhythm: float = Field(default=1.0, gt=0, le=16)
    notes: list[NoteInput] = Field(min_length=1, max_length=128)

    @model_validator(mode='after')
    def check_timing(self):
        numerator, denominator = self.time_signature
        if not (1 <= numerator <= 32 and denominator in (1, 2, 4, 8, 16)):
            raise ValueError('Unsupported time signature')
        if sum(n.duration for n in self.notes) > 256:
            raise ValueError('Melody must be at most 256 quarter-note beats')
        if sum(math.ceil(n.duration / self.harmonic_rhythm) for n in self.notes) > 256:
            raise ValueError('Harmonic rhythm would exceed 256 slots')
        return self

    def to_schema(self) -> Melody:
        return Melody(title=self.title, tonic=self.tonic, mode=self.mode,
                      time_signature=self.time_signature,
                      harmonic_rhythm=self.harmonic_rhythm,
                      notes=[Note(n.midi, n.duration) for n in self.notes])


class RunOptions(Model):
    backend: Literal['mock', 'aer', 'noisy', 'ibm'] = 'mock'
    solver: Literal['qaoa', 'xy_qaoa'] = 'qaoa'
    shots: int = Field(default=1024, ge=32, le=8192, strict=True)
    reps: int = Field(default=1, ge=1, le=5, strict=True)
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)
    window_size: int = Field(default=3, ge=1, le=4, strict=True)
    max_qubits: int = Field(default=16, ge=1, le=20, strict=True)
    max_iterations: int = Field(default=50, ge=1, le=200, strict=True)


class HarmonizeRequest(Model):
    melody: MelodyInput
    tempo: int = Field(default=100, ge=30, le=240, strict=True)
    style: Literal['chorale', 'strict', 'jazz_loose'] = 'chorale'
    run: RunOptions = Field(default_factory=RunOptions)
    use_cache: bool = True
