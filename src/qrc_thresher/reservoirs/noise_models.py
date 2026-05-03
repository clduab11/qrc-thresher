"""Noise model plugin scaffolding for Qiskit Aer sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NoiseModelSpec:
    """Serializable descriptor for configured noise model."""

    name: str
    params: dict[str, float]


def build_depolarizing_noise_model(probability: float = 0.001) -> Any:
    """Build a depolarizing noise model for Aer backends.

    Returns a Qiskit NoiseModel when qiskit-aer is available, otherwise a
    serializable fallback descriptor that callers can log.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f'probability must be in [0, 1], got {probability}')

    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except Exception:
        return NoiseModelSpec(name='depolarizing', params={'probability': probability})

    model = NoiseModel()
    error_1q = depolarizing_error(probability, 1)
    error_2q = depolarizing_error(probability, 2)
    model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz'])
    model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return model


def build_relaxation_noise_model(
    t1: float = 50e3,
    t2: float = 70e3,
    gate_time_1q: float = 50.0,
    gate_time_2q: float = 300.0,
) -> Any:
    """Build a simple thermal-relaxation noise model.

    All time values are in nanoseconds.
    """
    for name, value in {
        't1': t1,
        't2': t2,
        'gate_time_1q': gate_time_1q,
        'gate_time_2q': gate_time_2q,
    }.items():
        if value <= 0:
            raise ValueError(f'{name} must be > 0, got {value}')

    try:
        from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
    except Exception:
        return NoiseModelSpec(
            name='relaxation',
            params={
                't1': t1,
                't2': t2,
                'gate_time_1q': gate_time_1q,
                'gate_time_2q': gate_time_2q,
            },
        )

    model = NoiseModel()
    error_1q = thermal_relaxation_error(t1=t1, t2=t2, time=gate_time_1q)
    error_2q = thermal_relaxation_error(t1=t1, t2=t2, time=gate_time_2q).tensor(error_1q)
    model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz'])
    model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return model
