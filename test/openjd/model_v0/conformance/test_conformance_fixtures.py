# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs openjd-specifications conformance fixtures against the pure-Python model.

``fixtures/`` mirrors the layout of ``conformance-tests/`` in
openjd-specifications (``<revision>/<EXTENSION>/job_templates/*.yaml``). It holds a
copy of the fixtures, so that this suite runs without a checkout of the
specification. Following the specification's naming convention, a template whose
name contains ``.invalid.`` must fail to decode, and every other template must
decode. A valid template whose Job Parameters all have defaults must also create
a Job, since some rules apply only to values resolved at job creation.

Templates are decoded with every extension this implementation supports allowed,
like ``openjd check``: whether a template may use an extension is up to its own
``extensions`` list.

When a checkout of openjd-specifications is available (set
``OPENJD_SPECIFICATIONS_DIR``, or put it next to this repository),
``test_fixtures_match_the_specification`` checks the copy is current. To refresh
the copy, copy the fixture directories from ``conformance-tests/`` over
``fixtures/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from openjd.model import (
    DecodeValidationError,
    create_job,
    decode_job_template,
    preprocess_job_parameters,
)
from openjd.model.v2023_09 import ExtensionName

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_ALL_EXTENSIONS = [e.value for e in ExtensionName]

# Fixture directories (relative to fixtures/) that this suite runs.
_FIXTURE_DIRS = ["2023-09/STEP_APPROVALS/job_templates"]


def _fixture_files() -> list[Path]:
    files: list[Path] = []
    for directory in _FIXTURE_DIRS:
        files.extend(sorted((_FIXTURES_DIR / directory).glob("*.yaml")))
    return files


def _fixture_id(path: Path) -> str:
    return path.relative_to(_FIXTURES_DIR).as_posix()


def _fixture_params() -> list[Any]:
    return [pytest.param(path, id=_fixture_id(path)) for path in _fixture_files()]


def _decode(template: dict[str, Any]) -> Any:
    return decode_job_template(template=template, supported_extensions=_ALL_EXTENSIONS)


def test_fixture_dirs_are_not_empty() -> None:
    # Guards against a moved or emptied fixture directory silently running nothing.
    for directory in _FIXTURE_DIRS:
        assert list((_FIXTURES_DIR / directory).glob("*.yaml")), directory


@pytest.mark.parametrize("path", _fixture_params())
def test_conformance_fixture(path: Path) -> None:
    # GIVEN
    template = yaml.safe_load(path.read_text(encoding="utf-8"))

    if ".invalid." in path.name:
        # THEN
        with pytest.raises(DecodeValidationError):
            _decode(template)
        return

    # WHEN
    job_template = _decode(template)

    # THEN
    definitions = template.get("parameterDefinitions") or []
    if all("default" in definition for definition in definitions):
        parameter_values = preprocess_job_parameters(
            job_template=job_template,
            job_parameter_values={},
            job_template_dir=path.parent,
            current_working_dir=path.parent,
            # Fixtures use absolute PATH defaults such as /mnt/out.
            allow_job_template_dir_walk_up=True,
        )
        create_job(job_template=job_template, job_parameter_values=parameter_values)


def _specification_dir() -> Path:
    configured = os.environ.get("OPENJD_SPECIFICATIONS_DIR")
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root.parent / "openjd-specifications"


def test_fixtures_match_the_specification() -> None:
    conformance_dir = _specification_dir() / "conformance-tests"
    if not conformance_dir.is_dir():
        pytest.skip(f"No openjd-specifications checkout at {conformance_dir.parent}")

    for directory in _FIXTURE_DIRS:
        ours = {p.name: p.read_bytes() for p in (_FIXTURES_DIR / directory).glob("*.yaml")}
        theirs = {p.name: p.read_bytes() for p in (conformance_dir / directory).glob("*.yaml")}
        assert sorted(ours) == sorted(theirs), directory
        stale = sorted(name for name in ours if ours[name] != theirs[name])
        assert not stale, f"{directory}: {stale}"
