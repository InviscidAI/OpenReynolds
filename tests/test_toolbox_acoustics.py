"""`openreynolds/toolbox/acoustics.py`.

Its parsing is ordinary Python and so is its arithmetic, which is the half that
matters: a decibel is a ratio of two numbers and every way of getting it wrong here
produces a figure that looks exactly like a right one. Two of these tests check
against something outside this repository rather than against the script's own
behaviour -- the textbook SPL of a 1 Pa RMS sine, and ISO 266's preferred
frequencies -- because a self-consistent dB scale that is 60 dB out is the failure
this script exists to prevent.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def acoustics():
    return load("acoustics")


CONTROL_DICT = """\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     {application};
deltaT          {step};
functions
{{
    probes
    {{
        type            probes;
        fields          (p);
    }}
}}
"""

PRESSURE_FIELD = """\
FoamFile {{ version 2.0; format ascii; class volScalarField; object p; }}
dimensions      [{dimensions}];
internalField   uniform 0;
"""

KINEMATIC = "0 2 -2 0 0 0 0"
STATIC = "1 -1 -2 0 0 0 0"


def tone(seconds: float, rate: float, frequency: float, rms: float) -> tuple:
    """A sine of a known RMS on an exact time axis, which is the point of using one."""
    times = np.arange(0.0, seconds, 1.0 / rate)
    return times, math.sqrt(2.0) * rms * np.sin(2.0 * math.pi * frequency * times)


def write_probe_file(path: Path, times, values) -> Path:
    """A `probes` function object's own file layout, header lines included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{t:.10g}\t{v:.10g}\n" for t, v in zip(times, values))
    path.write_text("# Probe 0 (0.25 0 0.01)\n"
                    "#       Probe             0\n"
                    "#        Time\n" + rows, encoding="utf-8")
    return path


def build_case(root: Path, times, values, application: str = "pimpleFoam",
               dimensions: str = KINEMATIC, step: str = "1e-4") -> Path:
    """A case directory with the three files this script reads and nothing else."""
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(parents=True, exist_ok=True)
    (root / "system" / "controlDict").write_text(
        CONTROL_DICT.format(application=application, step=step), encoding="utf-8")
    (root / "0" / "p").write_text(
        PRESSURE_FIELD.format(dimensions=dimensions), encoding="utf-8")
    write_probe_file(root / "postProcessing" / "probes" / "0" / "p", times, values)
    return root


# -- validated against something outside this repository -----------------------


def test_a_one_pascal_rms_sine_is_ninety_four_decibels(tmp_path, acoustics):
    """The textbook anchor: SPL = 20 log10(p_rms / 20 uPa).

    20 log10(1 / 20e-6) = 93.979 dB, which is the number any acoustics text gives
    for a 1 Pa RMS signal against the standard reference pressure for air (ISO 1683
    fixes that reference at 20 micropascal; the arithmetic is the definition of the
    decibel and not this script's opinion). A sine of unit RMS is fed in through a
    plain CSV declared `--pa`, so nothing but the level arithmetic is under test,
    and both the overall SPL and the one-third-octave band holding the tone have to
    land on it.
    """
    times, values = tone(2.0, 4000.0, 200.0, rms=1.0)
    series = tmp_path / "history.csv"
    series.write_text("Time,p\n" + "".join(f"{t:.10g},{v:.10g}\n"
                                           for t, v in zip(times, values)),
                      encoding="utf-8")

    found = acoustics.analyse(series, column="p", already_pa=True)
    assert found["spectrum"]["overall_spl"] == pytest.approx(93.979, abs=0.01)
    # Not exactly 1.0: the level is of the fluctuation about the record's own best-fit
    # line, and removing that line from a finite run of a sine takes a little with it.
    assert found["spectrum"]["rms"] == pytest.approx(1.0, rel=1e-5)

    bands = {entry["nominal"]: entry["level"] for entry in found["third_octave"]["bands"]}
    # 200 Hz is a preferred centre, so the tone sits in the middle of its own band
    # and that band holds essentially all of the record's power.
    assert bands[200] == pytest.approx(93.979, abs=0.05)
    assert found["third_octave"]["loudest"]["nominal"] == 200
    assert found["reference_pressure"] == 20e-6
    text = acoustics.report(found)
    assert "re 2e-05 Pa" in text
    # A file read on its own has no controlDict beside it, so the steady check that
    # the case path refuses on could not run, and the report says that rather than
    # leaving the reader to assume it did.
    assert "system/controlDict was not read" in text


def test_the_band_centres_are_the_iso_266_preferred_series(acoustics):
    """ISO 266:1997, Table 1 -- the preferred one-third-octave frequencies.

    20, 25, 31.5, 40 ... 16000, 20000 Hz, the R10 Renard series rounded the way the
    standard rounds it (31.5 for 31.62, 63 for 63.10, 125 for 125.9). The labels
    come out of `RENARD_10` scaled by the decade rather than a typed table, so this
    is the check that the formula reproduces the standard across four decades. The
    exact centres stay 1000 * 10^(n/10) and sit within 1% of their labels, the
    widest rounding in the series being 158.49 Hz called 160 Hz.
    """
    iso_266 = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
               630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
               10000, 12500, 16000, 20000]
    # Band -17 is the one labelled 20 Hz (its exact centre is 19.95) and band 13 the
    # one labelled 20 kHz, which is the audible span the standard tabulates.
    bands = [acoustics.band(index) for index in range(-17, 14)]
    assert [entry["nominal"] for entry in bands] == pytest.approx(iso_266)
    for entry in bands:
        assert entry["centre"] == pytest.approx(entry["nominal"], rel=0.01)
        # The edges are a sixth of an octave either side, so the bands tile without
        # a gap: a third of an octave from one band's lower edge to its upper.
        assert entry["upper"] / entry["lower"] == pytest.approx(2.0 ** (1.0 / 3.0))
    # And the range used over a spectrum is the bands whose exact centres fall in it.
    assert list(acoustics.band_index_range(19.95, 19953.0)) == list(range(-17, 14))


# -- the unit trap -------------------------------------------------------------


def test_a_kinematic_case_is_refused_by_name_without_a_density(tmp_path, acoustics):
    """An incompressible case's `p` is p/rho, and no density is guessed for it.

    The refusal has to name what it read and what would fix it, because the
    alternative -- multiplying by 1.225 because the fluid is probably air -- is a
    number nobody can tell from a measurement afterwards.
    """
    times, values = tone(1.0, 2000.0, 120.0, rms=2.0)
    case = build_case(tmp_path / "case", times, values, dimensions=KINEMATIC)

    with pytest.raises(acoustics.results.Unavailable) as refusal:
        acoustics.analyse(case)
    message = str(refusal.value)
    assert "kinematic" in message
    assert "--density" in message
    assert "20*log10(rho)" in message

    found = acoustics.analyse(case, density=1.225)
    assert found["convention"]["kind"] == "kinematic"
    assert found["convention"]["density"] == 1.225
    assert found["convention"]["dimensions"] == [0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]
    # Every dB figure in the report carries the convention and the density with it.
    text = acoustics.report(found)
    assert "kinematic p x rho = 1.225 kg/m3" in text
    for line in text.splitlines():
        if " dB " in line and "does not tell you" not in line:
            assert "rho = 1.225" in line or "re 2e-05 Pa" in line


def test_a_static_case_is_read_as_pascals_with_no_density(tmp_path, acoustics):
    """`dimensions [1 -1 -2 0 0 0 0]` is already Pa: a compressible or `p_rgh`-free
    case needs nothing, and saying so is half of why the dimensions are read at all."""
    times, values = tone(1.0, 2000.0, 120.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, dimensions=STATIC)

    found = acoustics.analyse(case)
    assert found["convention"]["kind"] == "static"
    assert found["convention"]["density"] is None
    assert found["spectrum"]["overall_spl"] == pytest.approx(93.979, abs=0.05)
    assert "no density applied" in found["convention"]["basis"]


@pytest.mark.parametrize("density, decibels", [(1.225, 1.7627), (1000.0, 60.0)])
def test_a_kinematic_field_read_as_pascals_is_wrong_by_twenty_log_rho(
        tmp_path, acoustics, density, decibels):
    """The size of the trap, asserted rather than asserted about.

    p_Pa = rho * p_kinematic, so the two readings differ by 20*log10(rho) exactly:
    1.7627 dB for air at 1.225 kg/m3 and 60.0 dB for water at 1000. (A brief for this
    work put the air figure at "about 62 dB". 20*log10(1.225) = 1.763 and
    20*log10(1225) = 61.763, so that figure is the 1.225-against-1225 slip rather
    than the density of air. The trap is real at every density -- a dB figure 1.76
    out is still a wrong dB figure -- and it is 60 dB wide in water, which is where
    quoting one does the most damage.)
    """
    times, values = tone(1.0, 2000.0, 125.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, dimensions=KINEMATIC)

    as_pascals = acoustics.analyse(case, already_pa=True)
    converted = acoustics.analyse(case, density=density)
    difference = converted["spectrum"]["overall_spl"] - as_pascals["spectrum"]["overall_spl"]
    assert difference == pytest.approx(20.0 * math.log10(density), abs=1e-9)
    assert difference == pytest.approx(decibels, abs=1e-3)
    # Band by band as well: a constant factor on the pressure is a constant offset
    # on every level, so nothing about the shape hides it.
    for untouched, right in zip(as_pascals["third_octave"]["bands"],
                                converted["third_octave"]["bands"]):
        assert right["level"] - untouched["level"] == pytest.approx(difference, abs=1e-6)


# -- the steady-run trap -------------------------------------------------------


def test_a_steady_run_gets_no_spectrum_and_no_wav(tmp_path, acoustics, capsys):
    """A SIMPLE-family run's `Time` is an iteration count, so it has no sound.

    This is the defect `spectra.py` carried -- a Strouhal number a hundredfold out
    because iterations were read as seconds -- and the consequence here would be a
    playable file at an invented pitch, which is worse than a wrong number because
    it sounds like evidence.
    """
    times, values = tone(1.0, 2000.0, 120.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, application="simpleFoam",
                      dimensions=STATIC)
    target = tmp_path / "steady.wav"

    assert acoustics.main([str(case), "--wav", str(target), "--json",
                           str(tmp_path / "steady.json")]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("not measured:")
    assert "simpleFoam" in printed
    assert "iterations" in printed
    assert "dB" not in printed
    assert not target.exists(), "a steady run must not leave a .wav behind"
    assert not (tmp_path / "steady.json").exists()
    assert acoustics.is_steady_solver("rhoSimpleFoam")
    assert not acoustics.is_steady_solver("pimpleFoam")


# -- the sound file ------------------------------------------------------------


def test_the_wav_is_a_readable_header_at_the_stated_scale(tmp_path, acoustics):
    """Written with the standard library's `wave`, so `wave` reads it back.

    The scale factor is the only route from the file back to pascals, so what is
    checked is that the number printed is the number applied: the loudest sample is
    the stated peak in pascals times the stated counts per pascal.
    """
    times, values = tone(0.5, 4000.0, 150.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, dimensions=STATIC)
    target = tmp_path / "sound.wav"

    found = acoustics.analyse(case)
    written = acoustics.write_wav(found["_arrays"]["t"], found["_arrays"]["pressure"],
                                 target, rate=22050, stretch=1.0)
    found["wav"] = written

    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 22050
        assert handle.getnframes() == written["frames"]
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")

    assert samples.size == written["frames"]
    assert written["calibrated"] is False
    # The stated factor is the one applied, so the loudest sample is the stated peak
    # in pascals times it. Never above: the audio grid is interpolated between the
    # record's own samples, so it can miss the top of a crest by a fraction of a
    # percent and cannot overshoot it.
    loudest = int(np.max(np.abs(samples)))
    assert loudest <= written["peak_pa"] * written["scale"]
    assert loudest == pytest.approx(written["peak_pa"] * written["scale"], rel=0.01)
    assert loudest == pytest.approx(acoustics.HEADROOM * acoustics.FULL_SCALE, rel=0.01)
    text = acoustics.report(found)
    assert f"{written['scale']:.6g} counts/Pa" in text
    assert "not calibrated sound" in text


def test_the_audio_and_physical_durations_both_appear_and_differ_by_the_stretch(
        tmp_path, acoustics):
    """Four physical tenths become four audio seconds at stretch 10, and the report
    says both numbers and the factor between them -- because the pitch moves with it
    and a listener has no way to know that from the file."""
    times, values = tone(0.4, 5000.0, 400.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, dimensions=STATIC)

    found = acoustics.analyse(case)
    written = acoustics.write_wav(found["_arrays"]["t"], found["_arrays"]["pressure"],
                                 tmp_path / "slow.wav", rate=44100, stretch=10.0)
    found["wav"] = written

    physical = written["physical_seconds"]
    assert physical == pytest.approx(0.4 - 1.0 / 5000.0, rel=1e-9)
    assert written["seconds"] == pytest.approx(physical * 10.0, rel=1e-4)
    assert written["playback_rate"] == pytest.approx(written["record_rate"] / 10.0)

    text = acoustics.report(found)
    assert f"{written['seconds']:.4g} audio seconds" in text
    assert f"{physical:.4g} s record" in text
    assert "stretch 10x" in text
    assert "divided by 10" in text
    assert "only right if the time column was" in text


# -- the record's own limits ---------------------------------------------------


def test_a_record_too_short_for_the_lowest_band_asked_for_says_so(tmp_path, acoustics):
    """A band narrower than the spectrum's resolution holds no bin at all.

    A one-third-octave band is 23.2% of its centre wide, so at 25 Hz resolution the
    100 Hz band (89.1 to 112.2 Hz) cannot be formed -- and a band reported as 0 dB
    because nothing landed in it reads exactly like a quiet one. What comes back is
    the record length that would resolve it instead of a number.
    """
    times, values = tone(0.1, 1000.0, 300.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, dimensions=STATIC)

    found = acoustics.analyse(case, lowest_band=100.0)
    asked = found["third_octave"]["asked"]
    assert asked["nominal"] == 100
    assert asked["resolved"] is False
    assert asked["needs_span"] > found["record"]["span"]
    assert all(entry["nominal"] != 100 for entry in found["third_octave"]["bands"])

    text = acoustics.report(found)
    assert "the lowest band asked for, 100 Hz, is not resolved" in text
    assert f"{asked['needs_span']:.4g} s" in text
    # And the bands that are resolved are still reported, with the 300 Hz tone in
    # the 315 Hz band: a refusal about one band is not a refusal about the record.
    assert found["third_octave"]["loudest"]["nominal"] == 315


def test_the_probe_columns_carry_where_they_were_sampled(tmp_path, acoustics):
    """`results.parse_dat` drops the `# Probe 0 (x y z)` lines and the columns come
    out `c1`, `c2`. A near-field level is a level somewhere, so the coordinates are
    read back and the column named for the probe it belongs to."""
    times, values = tone(0.5, 2000.0, 120.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, dimensions=STATIC)

    found = acoustics.analyse(case)
    assert found["column"] == "probe0"
    assert found["probes"] == {"0": [0.25, 0.0, 0.01]}
    assert "sampled at (0.25 0 0.01)" in acoustics.report(found)


def test_what_it_cannot_do_is_in_the_report_and_the_case_is_probed(tmp_path, acoustics):
    """A near-field history is not a far-field prediction, and the report says so
    beside the numbers rather than in a docstring nobody reads. What the case already
    ran is probed on disk, so a real FWH or `noise` result is named as the one to
    quote instead of this."""
    times, values = tone(0.5, 2000.0, 120.0, rms=1.0)
    case = build_case(tmp_path / "case", times, values, dimensions=STATIC)
    (case / "postProcessing" / "noise").mkdir(parents=True)
    (case / "system" / "noiseDict").write_text("FoamFile { object noiseDict; }\n",
                                               encoding="utf-8")

    found = acoustics.analyse(case)
    text = acoustics.report(found)
    assert "Ffowcs Williams-Hawkings" in text
    assert "not a propagated far-field prediction" in text
    assert "nothing here recomputes it" in text
    assert str(case / "postProcessing" / "noise") in text
    assert any(path.endswith("noiseDict") for path in found["far_field"]["dictionaries"])
    assert {"name": "probes", "type": "probes", "file": "controlDict",
            "role": "pressure history"} in found["far_field"]["objects"]
    # The probe is a probe: a tool that is not on this machine is an answer, not an
    # error, which is `cfmesh.py:where`'s arrangement.
    assert "noise" in found["far_field"]["utilities"]


def test_the_cli_runs_end_to_end_and_exits_zero(tmp_path, acoustics, capsys):
    """One pass through `main` with every output asked for, because the flags are
    where a working library goes wrong, and exit 0 is the toolbox's contract."""
    times, values = tone(0.5, 4000.0, 250.0, rms=2.0)
    case = build_case(tmp_path / "case", times, values, dimensions=KINEMATIC)

    code = acoustics.main([str(case), "--density", "1.225", "--stretch", "4",
                           "--wav", str(tmp_path / "out.wav"),
                           "--png", str(tmp_path / "out.png"),
                           "--json", str(tmp_path / "out.json")])
    printed = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "out.wav").is_file()
    assert (tmp_path / "out.png").is_file()
    assert (tmp_path / "out.json").is_file()
    assert "overall SPL:" in printed and "one-third-octave" in printed

    import json as _json
    payload = _json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["convention"]["density"] == 1.225
    assert payload["wav"]["stretch"] == 4.0
    assert "_arrays" not in payload
    # 2 m2/s2 RMS at 1.225 kg/m3 is 2.45 Pa RMS, which is 101.77 dB.
    assert payload["spectrum"]["overall_spl"] == pytest.approx(
        20.0 * math.log10(2.0 * 1.225 / 20e-6), abs=0.05)


def test_an_empty_or_flat_history_is_a_reading_and_not_a_crash(tmp_path, acoustics):
    """A constant pressure has no sound in it, and 10*log10(0) is not a number to
    print. Both halves of that are refusals with the reason attached."""
    times = np.arange(0.0, 0.5, 1.0 / 2000.0)
    case = build_case(tmp_path / "case", times, np.full(times.size, 101325.0),
                      dimensions=STATIC)
    with pytest.raises(acoustics.results.Unavailable) as refusal:
        acoustics.analyse(case)
    assert "does not vary" in str(refusal.value)

    bare = build_case(tmp_path / "short", times[:4], np.zeros(4), dimensions=STATIC)
    with pytest.raises(acoustics.results.Unavailable) as too_short:
        acoustics.analyse(bare)
    assert "under the 16" in str(too_short.value)
