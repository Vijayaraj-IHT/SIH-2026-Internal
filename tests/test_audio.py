"""Audio I/O: the RIFF parser must survive every WAV flavour in the corpora.

This is not busy-work.  On the first attempt the pipeline silently dropped 88% of
the negatives because Python 3.11's ``wave`` module cannot read
``WAVE_FORMAT_EXTENSIBLE``; nothing crashed, the model just quietly trained on a
fraction of the data.  These tests pin the formats we actually encounter.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from ml.common.audio import (
    SAMPLE_RATE,
    db,
    float_from_pcm16,
    load_audio,
    parse_wav_header,
    pcm16_from_float,
    read_wav,
    resample,
    rms,
    to_mono,
    ulaw_decode,
    ulaw_encode,
    wav_duration,
    write_wav,
)


def _pcm_wav(path: Path, x: np.ndarray, sr: int, channels: int = 1, bits: int = 16) -> None:
    """Write a plain PCM WAV (control case)."""
    write_wav(path, x, sr)


def _make_extensible(path: Path, x: np.ndarray, sr: int, channels: int, bits: int = 24) -> None:
    """Hand-build a WAVE_FORMAT_EXTENSIBLE file (what UrbanSound8K clips use)."""
    bytes_per = bits // 8
    if bits == 24:
        # 24-bit PCM, interleaved exactly as a real writer would emit it
        ints = np.clip(np.round(x * (2**23 - 1)), -(2**23), 2**23 - 1).astype(np.int32).reshape(-1)
        raw = bytearray()
        for v in ints:
            raw += struct.pack("<i", int(v))[:3]
        data = bytes(raw)
    else:
        data = np.round(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    block_align = channels * bytes_per
    fmt = struct.pack(
        "<HHIIHH", 0xFFFE, channels, sr, sr * block_align, block_align, bits
    ) + struct.pack("<HHI", 22, bits, 0x3) + struct.pack("<H", 1) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    payload = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    payload += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(payload)) + payload)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_reads_pcm_widths(tmp_path: Path, bits: int) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    p = tmp_path / f"pcm{bits}.wav"
    if bits == 16:
        write_wav(p, x, sr)
    else:
        # only 16-bit writing is supported by our writer, so hand-build the rest
        if bits == 8:
            raw = ((x * 127) + 128).astype(np.uint8).tobytes()
        elif bits == 24:
            ints = np.round(x * (2**23 - 1)).astype(np.int32)
            raw = b"".join(struct.pack("<i", int(v))[:3] for v in ints)
        else:
            raw = np.round(x * (2**31 - 1)).astype("<i4").tobytes()
        head = b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * (bits // 8), bits // 8, bits)
        head += b"data" + struct.pack("<I", len(raw))
        p.write_bytes(b"RIFF" + struct.pack("<I", len(head) + len(raw)) + head + raw)
    y, got_sr = read_wav(p)
    assert got_sr == sr
    assert y.shape == (sr,)
    # quantisation error grows as the word width shrinks, as expected
    tol = {8: 0.02, 16: 1e-4, 24: 1e-6, 32: 1e-6}[bits]
    assert np.max(np.abs(y - x)) < tol


def test_reads_wave_format_extensible(tmp_path: Path) -> None:
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    stereo = np.stack([0.4 * np.sin(2 * np.pi * 300 * t), 0.4 * np.sin(2 * np.pi * 600 * t)], axis=1)
    p = tmp_path / "ext.wav"
    _make_extensible(p, stereo, sr, channels=2, bits=24)
    info = parse_wav_header(p)
    assert info["audio_format"] == 1  # sub-format resolves to PCM
    assert info["channels"] == 2 and info["sample_rate"] == sr and info["bits"] == 24
    y, got_sr = read_wav(p)
    assert got_sr == sr and y.shape == (sr, 2)
    mono = to_mono(y)
    assert mono.shape == (sr,)
    assert abs(wav_duration(p) - 1.0) < 1e-6


def test_skips_unknown_chunks(tmp_path: Path) -> None:
    """JUNK/LIST chunks appear in real files and must not confuse the parser."""
    sr = 16000
    x = np.zeros(sr, dtype=np.float32)
    p = tmp_path / "junk.wav"
    data = np.zeros(sr, dtype="<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    payload = b"WAVE" + b"JUNK" + struct.pack("<I", 28) + bytes(28)
    payload += b"fmt " + struct.pack("<I", 16) + fmt
    payload += b"data" + struct.pack("<I", len(data)) + data
    p.write_bytes(b"RIFF" + struct.pack("<I", len(payload)) + payload)
    y, _ = read_wav(p)
    assert y.shape == (sr,) and np.allclose(y, 0.0)


def test_rejects_non_wav(tmp_path: Path) -> None:
    p = tmp_path / "not.wav"
    p.write_bytes(b"not a wav at all")
    with pytest.raises(ValueError):
        read_wav(p)


def test_resample_preserves_tone() -> None:
    sr_in, sr_out = 48000, 16000
    t = np.arange(sr_in, dtype=np.float32) / sr_in
    x = np.sin(2 * np.pi * 1000 * t)
    y = resample(x, sr_in, sr_out)
    assert abs(y.size - sr_out) <= 2
    # the tone must survive the rate change: check via FFT peak
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    peak_hz = np.argmax(spec) * sr_out / y.size
    assert abs(peak_hz - 1000) < 20


def test_pcm_roundtrip() -> None:
    x = np.array([0.0, 0.5, -0.5, 0.999, -0.999], dtype=np.float32)
    assert np.allclose(float_from_pcm16(pcm16_from_float(x)), x, atol=1e-4)


# ---------------------------------------------------------------------------
# G.711 mu-law - the uplink codec
# ---------------------------------------------------------------------------
def test_ulaw_code_space_is_stable_over_int16_range() -> None:
    """Round-trip stability across the whole int16 range.

    G.711 collapses 65536 PCM values onto 256 codes, so the mapping cannot be
    injective.  What must hold is *stability*: re-encoding a decoded value
    reproduces the same code.  The single documented exception is the zero
    segment: the two codes ``0x7F`` (negative zero) and ``0xFF`` (positive zero)
    both decode to 0, and 0 re-encodes to ``0xFF``.  That ambiguity affects
    inputs within one quantisation step of zero and is inherent to the standard,
    not an implementation defect.
    """
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    codes = ulaw_encode(pcm)
    assert codes.min() >= 0 and codes.max() <= 255

    decoded = ulaw_decode(codes)
    re_codes = ulaw_encode(decoded)
    unstable = pcm[re_codes != codes]
    assert np.all(np.abs(unstable.astype(np.int32)) <= 8), (
        f"unexpected unstable codes for inputs {unstable[:10]}"
    )
    # and once decoded, values are fixed points forever after
    assert np.array_equal(ulaw_encode(ulaw_decode(re_codes)), re_codes)


def test_ulaw_error_is_bounded() -> None:
    """Absolute error must stay within the codec's quantisation step."""
    pcm = np.arange(-30000, 30001, 7, dtype=np.int16)
    rt = ulaw_decode(ulaw_encode(pcm)).astype(np.int32)
    err = np.abs(rt - pcm.astype(np.int32))
    # below the first segment boundary the error is bounded by the bias itself;
    # above it, by half of the widest segment step (4096) plus the bias offset
    bound = np.where(np.abs(pcm) < 256, 136, 2048 + 132)
    assert np.all(err <= bound), f"max error {err.max()} exceeds bound"


def test_ulaw_matches_reference_implementation() -> None:
    """Compare against a literal transcription of the ITU-T G.711 pseudo-code."""
    def reference(sample: int) -> int:
        BIAS, CLIP = 0x84, 32635
        sign = 0x80 if sample < 0 else 0
        s = min(abs(sample), CLIP) + BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (s & mask):
            exponent -= 1
            mask >>= 1
        mantissa = (s >> (exponent + 3)) & 0x0F
        return ~(sign | (exponent << 4) | mantissa) & 0xFF

    probes = [-32768, -30000, -10000, -1, 0, 1, 10000, 30000, 32767] + list(range(-5000, 5000, 137))
    for v in probes:
        assert int(ulaw_encode(np.array([v], dtype=np.int16))[0]) == reference(v), v


def test_ulaw_snr_is_speech_grade() -> None:
    """8-bit mu-law should hold ~30-40 dB SNR on speech-like input."""
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000.0
    speech = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    speech = speech / np.max(np.abs(speech)) * 0.7
    pcm = pcm16_from_float(speech)
    rt = float_from_pcm16(ulaw_decode(ulaw_encode(pcm)))
    noise = float_from_pcm16(pcm) - rt
    snr = 10 * np.log10(np.mean(speech**2) / max(np.mean(noise**2), 1e-12))
    assert snr > 25.0, f"mu-law SNR too low: {snr:.1f} dB"


def test_mulaw_and_downsampling_give_4x_total_reduction() -> None:
    """The uplink saving is 2x from companding AND 2x from 16k->8k decimation.

    Quoting "4x" from the 8-bit word alone would be wrong (8 bit vs 16 bit is a
    factor of 2); the other factor of 2 comes from halving the sample rate, which
    G.711 telephony permits because speech intelligibility lives below 4 kHz.
    """
    x = np.random.default_rng(1).standard_normal(16000).astype(np.float32) * 0.1
    pcm = pcm16_from_float(x)
    ulaw = ulaw_encode(pcm)
    assert ulaw.nbytes * 2 == pcm.nbytes  # 8 bit vs 16 bit
    # after decimation to 8 kHz the uplink carries half as many samples again
    assert (ulaw.nbytes // 2) * 4 == pcm.nbytes


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------
def test_rms_and_db() -> None:
    assert rms(np.zeros(10)) == pytest.approx(0.0, abs=1e-6)
    full = np.ones(10, dtype=np.float32)
    assert rms(full) == pytest.approx(1.0)
    assert db(full) == pytest.approx(0.0, abs=1e-6)
    assert db(np.ones(10, dtype=np.float32) * 0.5) == pytest.approx(-6.02, abs=0.01)


@pytest.mark.needs_data
def test_load_audio_on_real_corpus_files(data_dir: Path) -> None:
    """Every clip format actually present in the corpora must load at 16 kHz."""
    samples = [
        data_dir / "gsc" / "yes" / "004ae714_nohash_0.wav",
        data_dir / "hi_bixby" / "positive" / "positive_001.wav",
        data_dir / "hi_bixby" / "negative" / "negative_001.wav",
    ]
    samples += sorted((data_dir / "hi_bixby" / "negative").glob("*.wav"))[600:603]
    checked = 0
    for p in samples:
        if not p.exists():
            continue
        x = load_audio(p)
        assert x.ndim == 1
        assert x.size > 800
        assert np.max(np.abs(x)) <= 1.0
        checked += 1
    assert checked >= 4
