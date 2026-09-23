"""Pre-registration guards (docs/DECISIONS.md D005, D008).

A pre-registered protocol is frozen: these tests fail if configs/gates/G0.7.v1.yaml changes.
A changed protocol must be a new version file with its own DECISIONS entry, never an edit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
G07_V1 = REPO_ROOT / 'configs' / 'gates' / 'G0.7.v1.yaml'

# SHA-256 of the canonical JSON of the parsed YAML (sorted keys), so line endings and
# comments do not change it. Pinned when v1 was registered.
G07_V1_SHA256 = '969b9482e26cb62c4698f797c333e7bdd9030d35fccf272ee2a86ace5a3eafda'


def _registered() -> dict:
    return yaml.safe_load(G07_V1.read_text(encoding='utf-8'))


def test_g07_v1_is_frozen() -> None:
    canonical = json.dumps(_registered(), sort_keys=True, ensure_ascii=True)
    assert hashlib.sha256(canonical.encode()).hexdigest() == G07_V1_SHA256


def test_g07_v1_holds_the_registered_values() -> None:
    reg = _registered()
    assert (reg['gate'], reg['version'], reg['option']) == ('G0.7', 1, 'A')
    assert reg['measurement'] == {'model': 'exact', 'label': 'exact (oracle upper bound)'}
    assert reg['seeds']['min_seeds'] == 3
    assert reg['seeds']['rule'] == 'every_seed'

    stm = reg['stm']
    assert (stm['length'], stm['train_frac'], stm['delay_max']) == (500, 0.7, 20)
    assert stm['washout'] == 50
    assert stm['significance_level'] == 0.05

    parity = reg['parity']
    assert (parity['window'], parity['length'], parity['train_frac']) == (2, 500, 0.7)
    assert parity['washout'] == 50
    assert parity['significance_level'] == 0.05

    null = reg['permutation_null']
    assert null['n_permutations'] == 200
    assert null['permute'] == 'train_and_test_independently'
    assert null['refit_readout'] is True

    assert reg['readout']['alpha_selection'] == 'joint'
    assert reg['readout']['cv_scheme'] == 'contiguous'


def test_g07_v1_readout_matches_the_default_config(default_config) -> None:
    reg = _registered()
    assert reg['readout']['ridge_alphas'] == default_config.training.ridge_alphas
    assert reg['readout']['cv_folds'] == default_config.training.cv_folds
