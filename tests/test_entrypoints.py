# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry-point, discovery, and compliance-suite integration tests.

These confirm the plugin is discoverable and protocol-compliant the way an
application (and CI) sees it — they exercise the *installed* entry points via
``standard_asr.discover_models`` and ``standard_asr.compliance``. This is the
"one engine, many models" surface: a single engine_id with five model keys.
"""

from __future__ import annotations

from standard_asr import discover_models
from standard_asr.audio_format import AudioFormat
from standard_asr.compliance import (
    check_entrypoints,
    check_streaming_param_gating,
    check_sync_bridge,
)

from std_mlx_audio import (
    ParakeetTdt06BV3,
    Qwen3Asr06B,
    Qwen3Asr17B,
    WhisperLargeV3Turbo,
    WhisperTiny,
    create_canary_1b_v2,
    create_cohere_asr,
    create_fireredasr2_aed,
    create_fun_asr_nano,
    create_glm_asr_nano,
    create_granite_speech_1b,
    create_granite_speech_nar_2b,
    create_mms_1b_all,
    create_moonshine_tiny,
    create_nemotron_asr_streaming_0_6b,
    create_parakeet_tdt_0_6b_v3,
    create_qwen2_audio_7b,
    create_qwen3_asr_0_6b,
    create_qwen3_asr_1_7b,
    create_sensevoice_small,
    create_vibevoice_asr,
    create_voxtral_mini_3b,
    create_voxtral_realtime_4b,
    create_whisper_large_v3_turbo,
    create_whisper_tiny,
)

#: Every entry-point factory paired with the model key it must build. Calling each
#: covers all factory bodies and pins the factory<->entry-point-key wiring.
_FACTORY_KEYS = {
    create_qwen3_asr_0_6b: "mlx-audio/qwen3-asr-0.6b",
    create_qwen3_asr_1_7b: "mlx-audio/qwen3-asr-1.7b",
    create_parakeet_tdt_0_6b_v3: "mlx-audio/parakeet-tdt-0.6b-v3",
    create_whisper_large_v3_turbo: "mlx-audio/whisper-large-v3-turbo",
    create_whisper_tiny: "mlx-audio/whisper-tiny",
    create_nemotron_asr_streaming_0_6b: "mlx-audio/nemotron-asr-streaming-0.6b",
    create_sensevoice_small: "mlx-audio/sensevoice-small",
    create_cohere_asr: "mlx-audio/cohere-asr",
    create_fun_asr_nano: "mlx-audio/fun-asr-nano",
    create_voxtral_mini_3b: "mlx-audio/voxtral-mini-3b",
    create_canary_1b_v2: "mlx-audio/canary-1b-v2",
    create_qwen2_audio_7b: "mlx-audio/qwen2-audio-7b",
    create_glm_asr_nano: "mlx-audio/glm-asr-nano",
    create_granite_speech_1b: "mlx-audio/granite-speech-1b",
    create_granite_speech_nar_2b: "mlx-audio/granite-speech-nar-2b",
    create_vibevoice_asr: "mlx-audio/vibevoice-asr",
    create_moonshine_tiny: "mlx-audio/moonshine-tiny",
    create_mms_1b_all: "mlx-audio/mms-1b-all",
    create_fireredasr2_aed: "mlx-audio/fireredasr2-aed",
    create_voxtral_realtime_4b: "mlx-audio/voxtral-realtime-4b",
}

_EXPECTED_KEYS = {
    "mlx-audio/qwen3-asr-0.6b",
    "mlx-audio/qwen3-asr-1.7b",
    "mlx-audio/parakeet-tdt-0.6b-v3",
    "mlx-audio/whisper-large-v3-turbo",
    "mlx-audio/whisper-tiny",
    "mlx-audio/nemotron-asr-streaming-0.6b",
    "mlx-audio/sensevoice-small",
    "mlx-audio/cohere-asr",
    "mlx-audio/fun-asr-nano",
    "mlx-audio/voxtral-mini-3b",
    "mlx-audio/canary-1b-v2",
    "mlx-audio/qwen2-audio-7b",
    "mlx-audio/glm-asr-nano",
    "mlx-audio/granite-speech-1b",
    "mlx-audio/granite-speech-nar-2b",
    "mlx-audio/vibevoice-asr",
    "mlx-audio/moonshine-tiny",
    "mlx-audio/mms-1b-all",
    "mlx-audio/fireredasr2-aed",
    "mlx-audio/voxtral-realtime-4b",
}


def test_all_presets_discovered() -> None:
    names = set(discover_models().names())
    assert names >= _EXPECTED_KEYS


def test_by_engine_lists_all_presets_under_one_engine() -> None:
    # The whole point of this plugin: one engine_id, many models.
    assert set(discover_models().keys_by_engine("mlx-audio")) == _EXPECTED_KEYS


def test_registry_resolves_engine_class_without_instantiation() -> None:
    registry = discover_models()
    assert registry.engine_class("mlx-audio/qwen3-asr-0.6b") is Qwen3Asr06B
    assert registry.engine_class("mlx-audio/parakeet-tdt-0.6b-v3") is ParakeetTdt06BV3
    # model_id must equal the registered key (compliance invariant).
    for key in _EXPECTED_KEYS:
        assert registry.engine_class(key).properties.model_id == key


def test_spec_components_parsed() -> None:
    spec = discover_models().spec("mlx-audio/qwen3-asr-0.6b")
    assert spec.engine_id == "mlx-audio"
    assert spec.model_name == "qwen3-asr-0.6b"


def test_create_via_registry() -> None:
    assert isinstance(discover_models().create("mlx-audio/qwen3-asr-0.6b"), Qwen3Asr06B)


def test_check_entrypoints_passes() -> None:
    report = check_entrypoints()
    assert report.passed, [f"{i.level} {i.model} {i.message}" for i in report.issues]


def test_preset_factories_return_their_concrete_classes() -> None:
    assert type(create_qwen3_asr_0_6b()) is Qwen3Asr06B
    assert type(create_qwen3_asr_1_7b()) is Qwen3Asr17B
    assert type(create_parakeet_tdt_0_6b_v3()) is ParakeetTdt06BV3
    assert type(create_whisper_large_v3_turbo()) is WhisperLargeV3Turbo
    assert type(create_whisper_tiny()) is WhisperTiny


def test_every_factory_builds_its_keyed_preset() -> None:
    # Calls every entry-point factory (covering each body) and checks each builds a
    # preset whose model_id equals its registered key — the factory/key contract.
    keys = {factory(): key for factory, key in _FACTORY_KEYS.items()}
    assert {inst.properties.model_id for inst in keys} == set(_FACTORY_KEYS.values())
    for inst, key in keys.items():
        assert inst.properties.model_id == key


def test_streaming_param_gating_compliant_each_model() -> None:
    # Every model's streaming session must gate an unsupported standard parameter
    # per its strict/best_effort policy (compliance surface).
    for factory in (
        create_qwen3_asr_0_6b,
        create_parakeet_tdt_0_6b_v3,
        create_whisper_tiny,
    ):
        report = check_streaming_param_gating(factory())
        assert report.passed, [i.message for i in report.issues]


def test_sync_bridge_no_deadlock_each_family(fake_loader: object) -> None:
    # The standard sync->async bridge must terminate without deadlock or a leaked
    # thread for each backend family's windowed session.
    from .conftest import install_fake_loader  # noqa: F401 - fixture already patched

    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    for engine in (Qwen3Asr06B(), ParakeetTdt06BV3(), WhisperTiny()):
        report = check_sync_bridge(lambda e=engine: e.start_transcription(audio_format=fmt))
        assert report.passed, [i.message for i in report.issues]
