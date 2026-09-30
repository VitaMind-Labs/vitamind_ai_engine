"""Request contract for the Spark HTTP service.

Spark keeps its own planning schema (`OrganizeRequest`): the assistant, its
examples and its tests are written against it. The service wraps it in a small
envelope in the style of Lumina's `ContextBundle`:

    {"contract_version": "spark-contract-1",
     "track": "ADHD",                      <- resolved by the backend from its database
     "request": { ...OrganizeRequest... }}

`track` is the patient's authoritative support track as the backend resolved it
(clinician diagnosis first, then Mira orientation). It is the only thing Spark's
access rule reads. It is never taken from a patient, and never defaulted: the
inner `patient.supportTrack` defaults to ADHD, so leaning on it would make every
caller look like an ADHD patient.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from lumina.adhd.executive_function.schemas import OrganizeRequest

from . import CONTRACT_VERSION

Track = Literal["ADHD", "BIPOLAR", "SCHIZOPHRENIA", "UNSPECIFIED"]
# The one track Spark serves.
SPARK_TRACK = "ADHD"


class SparkOrganizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = CONTRACT_VERSION
    track: Track
    request: OrganizeRequest
