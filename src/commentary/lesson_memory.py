"""Small, explicit lesson memory for offline commentary experiments.

Lessons are model-written from a reviewed past failure. They are instructions,
not observations or evidence that an event happened in the next clip.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principle: str = Field(min_length=1, max_length=500)
    apply_when: str = Field(min_length=1, max_length=300)
    preserve: str = Field(min_length=1, max_length=300)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def with_lesson(system: str, lesson: Lesson | None) -> str:
    if lesson is None:
        return system
    return system + (
        "\nSaved lesson from a reviewed earlier call. This is guidance, not evidence "
        "about the current clip. Current evidence and role permissions take precedence.\n"
        + lesson.model_dump_json()
    )
