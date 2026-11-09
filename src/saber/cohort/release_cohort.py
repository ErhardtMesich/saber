"""Legacy cohort helpers; cohort assembly is disabled.

The original study's trajectories and paired responses are unavailable. Use the
real-data experiments documented in the README instead of generating replacements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from saber.cohort.registry import CorpusCatalogue
from saber.cohort.schema import CohortRecord, SpecimenRecord
from saber.runtime.atomic import write_json, write_text
from saber.runtime.config import PanelSpec, ResponseSpec, TwinSpec
from saber.twin.clock import SessionSchedule
from saber.twin.coupled import run_twin

Array = NDArray[np.float64]

@dataclass(frozen=True)
class ReleaseCohortSpec:
    """The parameters of the cohort: the article's printed counts and the session."""

    panel: PanelSpec
    response: ResponseSpec
    epochs: int
    epoch_hours: float
    release_seed: int

    @classmethod
    def from_configs(
        cls,
        panel: PanelSpec,
        response: ResponseSpec,
        twin: TwinSpec,
        seed: int,
    ) -> ReleaseCohortSpec:
        return cls(
            panel=panel,
            response=response,
            epochs=twin.epochs,
            epoch_hours=twin.epoch_hours,
            release_seed=seed,
        )

    def label(self) -> dict[str, object]:
        return {
            "corpora": dict(self.panel.corpora),
            "specimens": self.panel.specimens,
            "epochs": self.epochs,
            "epoch_hours": self.epoch_hours,
            "modality": self.panel.modality,
            "seed": self.release_seed,
        }


def assemble_release_cohort(spec: ReleaseCohortSpec, catalogue: CorpusCatalogue) -> CohortRecord:
    """Disable legacy experiment entry points without the original study records.

    The article's recorded trajectories, labels and response pairs are not
    available. A separately sourced image-only experiment is provided in
    scripts/run_brain_organoid_experiment.py; it does not invent these fields.
    """
    raise RuntimeError(
        "Historical cohort unavailable; legacy execution disabled. "
        "Use scripts/run_brain_organoid_growth.py with the documented real data table."
    )


def write_cohort(cohort: CohortRecord, root: object) -> tuple[str, str]:
    """Persist the cohort as a compressed array archive plus a JSON index.

    The archive holds the numeric arrays and the index holds the string fields and
    the shapes, so a reader can load either without parsing the other.
    """
    from pathlib import Path

    destination = Path(str(root))
    destination.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Array] = {}
    for specimen in cohort.specimens:
        arrays[f"axes/{specimen.identity}"] = specimen.axes
        arrays[f"observations/{specimen.identity}"] = specimen.observations
        arrays[f"dose_response/{specimen.identity}"] = specimen.dose_response
    archive = destination / "release_cohort.npz"
    writer: Any = np.savez_compressed
    writer(archive, **arrays)
    index = {
        "epoch_hours": cohort.epoch_hours,
        "dose_points": cohort.dose_points,
        "specimens": [specimen.label() for specimen in cohort.specimens],
    }
    index_path = destination / "cohort_index.json"
    write_json(index_path, index)
    write_text(
        destination / "corpus_counts.txt",
        "\n".join(f"{corpus} {len(cohort.by_corpus(corpus))}" for corpus in cohort.corpora())
        + "\n",
    )
    return (archive.name, index_path.name)


def read_cohort(root: object) -> CohortRecord:
    """Read back a persisted cohort."""
    from pathlib import Path

    destination = Path(str(root))
    index = json.loads((destination / "cohort_index.json").read_text(encoding="utf-8"))
    archive = np.load(destination / "release_cohort.npz")
    specimens = tuple(
        SpecimenRecord(
            identity=entry["identity"],
            corpus=entry["corpus"],
            stratum=entry["stratum"],
            transition_rate=float(entry["transition_rate"]),
            axes=np.asarray(archive[f"axes/{entry['identity']}"], dtype=np.float64),
            observations=np.asarray(archive[f"observations/{entry['identity']}"], dtype=np.float64),
            dose_response=np.asarray(
                archive[f"dose_response/{entry['identity']}"], dtype=np.float64
            ),
        )
        for entry in index["specimens"]
    )
    return CohortRecord(
        specimens=specimens,
        epoch_hours=float(index["epoch_hours"]),
        dose_points=int(index["dose_points"]),
    )


def release_cohort_twin_axes(
    spec: ReleaseCohortSpec, schedule: SessionSchedule, twin_spec: TwinSpec, seed: int
) -> Array:
    """The twin's own axis trajectory, used as the ground truth the twin supplies.

    Ref: Sec. 2.7 -- the twin is "the source of the ground truth policy value".
    """
    trajectory = run_twin(twin_spec, schedule, seed)
    return trajectory.axes_by_epoch()
