"""The public documents under docs/ carry no local path, no forbidden row column and no absolute
config_path (CP5 §10; docs/DECISIONS.md D019, ruling P4).

Covered: docs/scorecard.md, docs/AUDIT_2026-09.md, every text file under docs/evidence/** (.md,
.json, .csv, .sha256, .yaml, .gitattributes; the PNG is skipped) and docs/DECISIONS.md from the
D017 heading on. Each file is first asserted to exist (the run-4 folder with its full expected
content), so the module is red while the CP5b files are missing and never passes vacuously. Only
the generic patterns are checked here: a drive-letter path, a /Users/ or /home/ path, a JSON,
YAML or CSV-header key among the forbidden row columns, and an absolute config_path value.
"""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS = REPO_ROOT / 'docs'
EVIDENCE = DOCS / 'evidence'
RUN4 = EVIDENCE / '8ed2df4'
RUN4_FILES = [
    'MANIFEST.sha256', 'README.md', 'runs.sanitised.csv',
    'COMPARATIVE.v1.20260924T224106784228Z.json',
    'G1.20260924T224106784228Z.json', 'G2.20260924T224106784228Z.json',
    'G2.5.20260924T224106784228Z.json', 'G3.20260924T224106784228Z.json',
    'G4.20260924T224106784228Z.json',
    'G0.7.tuned_qrc.20260924T224100291861Z.json',
    'tuning/stm.json', 'tuning/parity.json', 'tuning/narma.json',
    'configs/comparative.yaml',
]
TEXT_SUFFIXES = {'.md', '.json', '.csv', '.sha256', '.yaml', '.gitattributes'}


def _assert_run4_folder() -> None:
    assert RUN4.is_dir(), 'docs/evidence/8ed2df4/ does not exist yet'
    missing = [name for name in RUN4_FILES if not (RUN4 / name).is_file()]
    assert missing == [], f'docs/evidence/8ed2df4/ lacks {missing}'


def _scan():
    return importlib.import_module('qrc_thresher.proof.evidence').scan_text


def _text_files_under_evidence() -> list:
    files = [p for p in EVIDENCE.rglob('*') if p.is_file()
             and (p.suffix in TEXT_SUFFIXES or p.name in TEXT_SUFFIXES)]
    assert files, 'docs/evidence/ holds no text file'
    return files


def _decisions_from_d017() -> str:
    text = (DOCS / 'DECISIONS.md').read_text(encoding='utf-8')
    marker = '## 2026-09-24: D017'
    assert marker in text, 'D017 is not in docs/DECISIONS.md'
    return text[text.index(marker):]


class TestPublicDocs:
    def test_the_scorecard_exists_and_is_clean(self) -> None:
        path = DOCS / 'scorecard.md'
        assert path.exists(), 'docs/scorecard.md does not exist yet'
        assert _scan()(path.read_text(encoding='utf-8')) == []

    def test_the_audit_exists_and_is_clean(self) -> None:
        path = DOCS / 'AUDIT_2026-09.md'
        assert path.exists(), 'docs/AUDIT_2026-09.md does not exist yet'
        assert _scan()(path.read_text(encoding='utf-8')) == []

    def test_every_evidence_text_file_is_clean(self) -> None:
        _assert_run4_folder()
        scan = _scan()
        hits = {}
        for path in _text_files_under_evidence():
            found = scan(path.read_text(encoding='utf-8'))
            if found:
                hits[path.relative_to(REPO_ROOT).as_posix()] = found
        assert hits == {}

    def test_the_gitattributes_pin(self) -> None:
        # A green guard once the file exists: git must store the evidence bytes as they are.
        path = EVIDENCE / '.gitattributes'
        assert path.exists(), 'docs/evidence/.gitattributes does not exist yet'
        assert path.read_bytes() == b'* -text\n'

    def test_decisions_from_d017_on_are_clean(self) -> None:
        assert _scan()(_decisions_from_d017()) == []
