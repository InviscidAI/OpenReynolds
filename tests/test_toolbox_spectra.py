"""`spectra.py`: the frequency and the settling of a time series.

Both answers are arithmetic on a series of numbers, so both are testable without
OpenFOAM, without a model key and without a network -- which is the whole reason
these live here rather than only in a study's notes. The signals are synthesised
with `numpy`, so every test knows the answer it is asking for.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"


def load(name: str, where: Path = TOOLBOX):
    """Import a script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"loaded_{name}", where / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spectra = load("spectra")


# -- fixtures ------------------------------------------------------------------


def sine(frequency: float, span: float = 20.0, samples: int = 2001,
         amplitude: float = 1.0, offset: float = 0.0, noise: float = 0.0,
         seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, span, samples)
    x = offset + amplitude * np.sin(2.0 * np.pi * frequency * t)
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(t.size)
    return t, x


def measure(t: np.ndarray, x: np.ndarray, segments: int = 4) -> dict:
    """The same pipeline `analyse` runs, on arrays instead of a file."""
    grid, values, _resampled = spectra.uniform_samples(t, x)
    spectrum = spectra.welch(grid, values, segments=segments)
    peak = spectra.dominant(spectrum, float(grid[-1] - grid[0]))
    verdict, basis = spectra.shedding_verdict(peak)
    return {"peak": peak, "verdict": verdict, "basis": basis,
            "settling": spectra.settle(grid, values, periods=peak["periods"]),
            "spectrum": spectrum}


def write_dat(path: Path, t: np.ndarray, columns: dict[str, np.ndarray]) -> Path:
    """A postProcessing `.dat` in OpenFOAM's layout: `#` meta lines, a `#` header
    naming the columns, then the rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Force coefficients", "# magUInf : 1.0", "# lRef : 0.01", "# Aref : 0.001",
             "# Time " + " ".join(columns)]
    for index, time in enumerate(t):
        lines.append(f"{time:.8g} " + " ".join(f"{values[index]:.8g}" for values in columns.values()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- the frequency -------------------------------------------------------------


def test_a_sine_of_known_frequency_comes_back_within_a_bin():
    """The floor of the whole script: if the transform's bins are mislabelled, or the
    window and the step are not paired, every number downstream of it is wrong by the
    same factor and nothing else here would notice."""
    found = measure(*sine(1.7))
    resolution = found["peak"]["resolution"]
    assert abs(found["peak"]["frequency"] - 1.7) < resolution
    assert found["verdict"] == "periodic"


def test_the_peak_survives_an_offset_a_drift_and_noise():
    """A real Cl history has a mean offset, a startup ramp and numerical noise on it.
    Without the per-segment linear detrend the ramp's power lands in the lowest bins
    -- where a slow shedding tone also lands -- and outbids the tone; that is the
    failure this protects against, not the noise."""
    t = np.linspace(0.0, 20.0, 2001)
    x = (0.4 + np.sin(2.0 * np.pi * 1.7 * t) + 0.03 * t
         + 0.3 * np.random.default_rng(5).standard_normal(t.size))
    found = measure(t, x)
    assert abs(found["peak"]["frequency"] - 1.7) < found["peak"]["resolution"]
    assert found["verdict"] == "periodic"


def test_the_spectrum_agrees_with_the_grader_zero_crossings():
    """`benchmarks/moving_mesh/grade.py:upward_crossings` was the only frequency
    measurement in the repository, and `fixed_cylinder` compares the Strouhal it
    builds from it against Williamson's 0.164. Two estimators of one quantity that
    disagree is how a study ends up with two Strouhal numbers and no way to tell
    which to believe, so they are pinned to each other here on the signal where both
    are valid: clean, zero-mean, many whole cycles."""
    grade = load("grade", ROOT / "benchmarks" / "moving_mesh")
    t, x = sine(1.7)
    crossings = grade.upward_crossings(t, x)
    assert len(crossings) >= 2
    by_crossings = (len(crossings) - 1) / (crossings[-1] - crossings[0])
    by_spectrum = measure(t, x)["peak"]["frequency"]
    assert abs(by_spectrum - by_crossings) < 0.5 * measure(t, x)["peak"]["resolution"]


def test_white_noise_is_aperiodic_rather_than_a_tone():
    """Every spectrum has a tallest bin, so `argmax` alone always finds a
    "frequency". A single periodogram of white noise has a tallest bin some nine
    times its own median -- the maximum of that many exponential draws -- which is
    why the default is four averaged segments and why the verdict is the prominence
    and not the peak. Several seeds, because passing on one is passing by luck."""
    for seed in range(6):
        t = np.linspace(0.0, 20.0, 2001)
        x = np.random.default_rng(seed).standard_normal(t.size)
        found = measure(t, x)
        assert found["verdict"] == "aperiodic", (
            f"seed {seed} gave prominence {found['peak']['prominence']:.2f}")
        assert found["peak"]["prominence"] < spectra.PROMINENCE_PERIODIC


def test_a_record_of_under_two_cycles_is_too_short_for_either_verdict():
    """A single cycle fits a sinusoid, a ramp and a step equally well, and the lowest
    resolvable bin of a record is one cycle per record -- so a "frequency" there can
    be the record's own length coming back. Both answers have to say `too-short`
    rather than one of them quoting a number off the same data."""
    t, x = sine(1.7, span=1.5 / 1.7, samples=200)
    found = measure(t, x)
    assert found["peak"]["periods"] < spectra.MIN_PERIODS
    assert found["verdict"] == "too-short"
    assert found["settling"]["verdict"] == "too-short"
    assert "cycles" in found["settling"]["basis"]


def test_the_frequency_is_refined_off_the_bin_it_was_found_in():
    """The parabolic correction, which exists because the bin width of a four-segment
    Welch over a 20 s record is 0.125 Hz and a Strouhal number quoted to three
    figures off that is three figures of the window length. 1.7 Hz falls between
    bins on purpose."""
    found = measure(*sine(1.7))
    assert found["peak"]["bin_frequency"] != found["peak"]["frequency"]
    assert abs(found["peak"]["frequency"] - 1.7) < 0.1 * abs(found["peak"]["bin_frequency"] - 1.7)


# -- the Strouhal number -------------------------------------------------------


def test_a_known_strouhal_is_recovered_from_a_synthetic_cl(tmp_path):
    """St = f L / U, on a Cl oscillating at a frequency chosen so that St is 0.2 for
    the L and U given. This is the number a cylinder study is graded on."""
    length, speed, target = 0.01, 1.0, 0.2
    frequency = target * speed / length
    t, x = sine(frequency, span=4.0, samples=4001)
    write_dat(tmp_path / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cd": np.full(t.size, 1.2), "Cl": x})
    found = spectra.analyse(tmp_path, column="Cl", length=length, speed=speed)
    assert found["strouhal"] is not None
    assert found["strouhal"]["value"] == pytest.approx(target, rel=0.02)


def test_no_length_or_speed_means_no_strouhal_number(tmp_path):
    """A diameter is not read off the mesh and a free-stream speed is not read off
    anything. `grade.py:_need` refuses rather than assumes, and a St invented from a
    default of 1.0 would be a dimensionless-looking number with no physics in it --
    the kind that gets compared against Williamson and reported as agreement."""
    t, x = sine(2.0, span=10.0, samples=1001)
    write_dat(tmp_path / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cd": np.full(t.size, 1.2), "Cl": x})
    for kwargs in ({}, {"length": 0.01}, {"speed": 1.0}):
        found = spectra.analyse(tmp_path, column="Cl", **kwargs)
        assert found["strouhal"] is None
    assert "not computed" in spectra.report(spectra.analyse(tmp_path, column="Cl"))


# -- settling ------------------------------------------------------------------


def test_a_drifting_series_is_still_moving_and_a_stationary_one_is_settled():
    """The distinction `results.py:tail_mean` deliberately refuses to make: it is the
    mean of the last 25% either way, and on a series still climbing it means nothing.
    A ramp and a limit cycle have to come out on opposite sides of it."""
    t = np.linspace(0.0, 20.0, 2001)
    noise = np.random.default_rng(3).standard_normal(t.size)

    ramp = spectra.settle(t, 0.1 * t + 0.01 * noise)
    assert ramp["verdict"] == "still-moving"
    assert ramp["index"] is None

    cycle = spectra.settle(t, 1.2 + 0.1 * np.sin(2.0 * np.pi * 1.7 * t) + 0.01 * noise)
    assert cycle["verdict"] == "settled"
    assert cycle["mean"] == pytest.approx(1.2, abs=0.01)


def test_the_stationary_index_skips_the_transient_and_keeps_the_rest():
    """A startup transient then a limit cycle, which is what a shedding case is. The
    index has to land in the cycle and not at the end of it: walking the candidate
    starts forwards and stopping at the first that passes reports the LONGEST usable
    window, where taking the last 25% by convention throws away most of the record
    it could have averaged.

    The mean is checked against the method's own resolution rather than against 1.2
    exactly: a half-against-half test can only see drift that is large beside the
    fluctuation it is drifting through, so the window it accepts still carries a
    residual transient of a fraction of the scatter (0.03 of a 0.07 standard
    deviation here). Asserting four figures would be asserting the seed."""
    t = np.linspace(0.0, 40.0, 4001)
    decay = 3.0 * np.exp(-t / 2.0)
    x = 1.2 + decay + 0.1 * np.sin(2.0 * np.pi * 1.7 * t)
    found = spectra.settle(t, x)
    assert found["verdict"] == "settled"
    assert 0.0 < found["from_time"] < 20.0
    assert abs(found["mean"] - 1.2) < 0.5 * found["std"]
    assert found["index"] + found["samples"] == t.size


def test_a_settling_transient_does_not_suppress_the_settling_answer():
    """This test exists because the first version got it backwards.

    A drag history is `1.2 + 3 exp(-t/2)` plus a small tone, which is the ordinary
    shape of a Cd. The per-segment detrend removes a line and an exponential is not
    a line, so the whole record's tallest bin is the transient -- 0.084 Hz, 1.67
    cycles of a 20 s record -- and settling, which asked the cycle count FIRST, came
    back `too-short` on a record that plainly contains a settled state. The cycle
    count now only speaks when no window passes the halves test, and the count a
    candidate window is judged by is measured on that window, where the real tone
    is."""
    t = np.linspace(0.0, 20.0, 4001)
    x = 1.2 + 3.0 * np.exp(-t / 2.0) + 0.05 * np.sin(2.0 * np.pi * 1.7 * t)
    found = measure(t, x)
    assert found["peak"]["periods"] < spectra.MIN_PERIODS  # the transient, not the flow
    assert found["verdict"] == "too-short"
    assert found["settling"]["verdict"] == "settled"
    assert found["settling"]["mean"] == pytest.approx(1.2, abs=0.03)


def test_cropping_the_transient_recovers_the_tone_the_whole_record_hid(tmp_path):
    """The other half of the same incident. `--from` is the only way the stationary
    time this script reports gets used, and without it a transient-dominated Cd has
    no route to a frequency at all."""
    t = np.linspace(0.0, 20.0, 4001)
    x = 1.2 + 3.0 * np.exp(-t / 2.0) + 0.05 * np.sin(2.0 * np.pi * 1.7 * t)
    write_dat(tmp_path / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cd": x})
    whole = spectra.analyse(tmp_path, column="Cd")
    assert whole["shedding"]["verdict"] == "too-short"

    cropped = spectra.analyse(tmp_path, column="Cd", start=8.0)
    assert cropped["record"]["cropped"]["dropped"] > 0
    assert cropped["shedding"]["verdict"] == "periodic"
    assert abs(cropped["peak"]["frequency"] - 1.7) < cropped["peak"]["resolution"]
    assert "--from 8 dropped" in spectra.report(cropped)


def test_a_record_too_small_to_have_halves_is_too_short():
    """Eight samples have a mean and a standard deviation and both are noise. The
    guard is on the sample count because the alternative is a confidence interval
    computed from two numbers and printed to four figures."""
    t = np.linspace(0.0, 1.0, 8)
    found = spectra.settle(t, np.linspace(1.0, 1.1, 8))
    assert found["verdict"] == "too-short"
    assert str(spectra.MIN_SAMPLES) in found["basis"]


def test_the_interval_widens_when_the_series_is_autocorrelated():
    """The reason the interval is on the effective sample size and not the row count.
    A force history is written every time step and consecutive rows agree to three
    figures; 20,000 correlated samples are not 20,000 measurements, and an interval
    built on the raw count is wrong by a large factor rather than a small one. Same
    variance, same length, one correlated and one not."""
    rng = np.random.default_rng(9)
    t = np.linspace(0.0, 20.0, 2001)
    white = rng.standard_normal(t.size)

    rho = 0.98
    shocks = rng.standard_normal(t.size)
    red = np.zeros_like(shocks)
    for index in range(1, shocks.size):
        red[index] = rho * red[index - 1] + shocks[index]
    red = red * np.sqrt(1.0 - rho ** 2)  # same variance as the white series

    independent = spectra.settle(t, white)
    correlated = spectra.settle(t, red)
    assert independent["verdict"] == correlated["verdict"] == "settled"
    assert independent["std"] == pytest.approx(correlated["std"], rel=0.3)
    assert correlated["integral_time"] > independent["integral_time"]
    assert correlated["independent"] < 0.2 * independent["independent"]
    assert correlated["ci95"] > 3.0 * independent["ci95"]


def test_the_integral_time_is_summed_only_to_the_first_zero_crossing():
    """Past the first crossing the autocorrelation is noise about zero, and summing
    it adds variance rather than information -- on white noise it would make the
    effective sample size a random number instead of the sample count."""
    white = np.random.default_rng(1).standard_normal(4000)
    found = spectra.integral_time(white, dt=0.01)
    assert found["tau"] == pytest.approx(0.0, abs=1e-3)
    assert found["independent"] == pytest.approx(4000, rel=0.01)


# -- reading the files ---------------------------------------------------------


def test_a_restarted_force_history_is_read_as_one_record(tmp_path):
    """A restart writes a second time directory holding a file of the same name, and
    both are one history. Reading only the newest loses the first half of the record
    -- which on a shedding case is the difference between eight cycles and three, and
    so between a measured frequency and `too-short`. The stitching is
    `results.read_history`'s rather than a second copy of it here."""
    frequency = 1.7
    base = tmp_path / "postProcessing" / "forceCoeffs"
    first_t = np.linspace(0.0, 12.0, 1201)
    second_t = np.linspace(10.0, 20.0, 1001)
    first = write_dat(base / "0" / "coefficient.dat", first_t,
                      {"Cd": np.full(first_t.size, 1.2),
                       "Cl": np.sin(2.0 * np.pi * frequency * first_t)})
    second = write_dat(base / "10" / "coefficient.dat", second_t,
                       {"Cd": np.full(second_t.size, 1.2),
                        "Cl": np.sin(2.0 * np.pi * frequency * second_t)})
    # `written_order` sorts a series by mtime, because the directory names of a run
    # restarted twice do not say which file its run is still writing.
    os.utime(first, (1_700_000_000, 1_700_000_000))
    os.utime(second, (1_700_000_100, 1_700_000_100))

    found = spectra.analyse(tmp_path, column="Cl")
    assert len(found["sources"]) == 2
    assert found["record"]["t0"] == pytest.approx(0.0)
    assert found["record"]["t1"] == pytest.approx(20.0)
    # The abandoned rows of the first file past t = 10 are dropped, not merged.
    assert found["record"]["samples"] < first_t.size + second_t.size
    assert abs(found["peak"]["frequency"] - frequency) < found["peak"]["resolution"]


def test_the_default_column_is_one_worth_drawing(tmp_path):
    """Column 1 of a `forces.dat` is the x component of the pressure force, which is
    not what anyone means by "the history". `results.plot_columns` already ranks the
    columns of a force file and its first choice is the default here."""
    t, x = sine(1.7, span=10.0, samples=1001)
    write_dat(tmp_path / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cd": np.full(t.size, 1.2), "Cl": x})
    assert spectra.analyse(tmp_path)["column"] == "Cd"


def test_a_csv_is_read_through_the_same_parser(tmp_path):
    """`results.parse_dat` already treats commas as separators and reads a
    `#`-commented header; a CSV's bare header row is the only difference, and it is
    marked as a comment rather than parsed by a second reader that would then have
    its own opinion about vector columns and half-written rows."""
    t, x = sine(1.7, span=10.0, samples=1001)
    path = tmp_path / "history.csv"
    path.write_text("Time,y\n" + "\n".join(f"{a:.8g},{b:.8g}" for a, b in zip(t, x)) + "\n",
                    encoding="utf-8")
    found = spectra.analyse(path, column="y")
    assert found["column"] == "y"
    assert abs(found["peak"]["frequency"] - 1.7) < found["peak"]["resolution"]


def test_a_varying_time_step_is_resampled_and_said_so(tmp_path):
    """An FFT assumes a constant step and a Courant-limited solver does not provide
    one. Interpolating onto the mean step is a distortion; reading the bin labels of
    a ragged series as frequencies is a wrong answer, so the resampling happens and
    the report states it."""
    t = np.sort(np.concatenate([np.linspace(0.0, 10.0, 600),
                                np.linspace(10.0, 20.0, 1400)[1:]]))
    x = np.sin(2.0 * np.pi * 1.7 * t)
    _grid, _values, resampled = spectra.uniform_samples(t, x)
    assert resampled
    found = measure(t, x)
    assert abs(found["peak"]["frequency"] - 1.7) < found["peak"]["resolution"]


def test_an_unmeasurable_case_reports_rather_than_raises(tmp_path):
    """Reporting scripts in this toolbox exit 0 whatever they find: not being able to
    measure something is a reading. A non-zero exit here would make an empty
    postProcessing directory look like a broken script."""
    assert spectra.main([str(tmp_path)]) == 0
    assert spectra.main([str(tmp_path / "nope.dat")]) == 0


# -- the command line ----------------------------------------------------------


def test_the_command_line_writes_the_report_the_figure_and_the_sidecar(tmp_path):
    """`--png` and `--json` are relative to where the script is RUN, not to the case
    (`ENVIRONMENT.md`): a caller who typed `--json spectra.json` from a study
    directory and found nothing where they stood is how that convention was earned
    in `mesh_look.py`."""
    t, x = sine(1.7, span=20.0, samples=2001)
    case = tmp_path / "case"
    write_dat(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cd": np.full(t.size, 1.2), "Cl": x})
    here = tmp_path / "ran_from"
    here.mkdir()

    done = subprocess.run(
        [sys.executable, str(TOOLBOX / "spectra.py"), str(case), "--column", "Cl",
         "--length", "0.01", "--speed", "1.0", "--png", "fft.png", "--json", "found.json"],
        cwd=here, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert "periodic" in done.stdout
    assert "St =" in done.stdout
    assert (here / "fft.png").is_file()

    found = json.loads((here / "found.json").read_text(encoding="utf-8"))
    assert found["shedding"]["verdict"] == "periodic"
    assert found["strouhal"]["value"] == pytest.approx(0.017, rel=0.05)
    assert "_arrays" not in found, "the sidecar carries findings, not the whole series"


def test_help_runs_without_a_case():
    """It is a standalone script run as `python3 spectra.py`, so `--help` has to work
    on a machine with no case, no OpenFOAM and nothing staged."""
    done = subprocess.run([sys.executable, str(TOOLBOX / "spectra.py"), "--help"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0
    for flag in ("--column", "--length", "--speed", "--png", "--json", "--segments", "--from"):
        assert flag in done.stdout


def test_the_report_prints_the_number_behind_every_judgement():
    """The house rule: `mesh_digest.py` applies no thresholds on purpose, and where
    this one does judge, the reader has to be able to disagree with the threshold
    without rerunning anything. So the prominence, the cycle count and the drift
    ratio appear in the text beside the verdict they produced."""
    t, x = sine(1.7, noise=0.3)
    found = measure(t, x)
    assert "x over the spectrum's median" in found["basis"] or "median" in found["basis"]
    assert f"{found['peak']['prominence']:.4g}" in found["basis"]
    assert "half-record standard deviations" in found["settling"]["basis"]


# -- the time axis -------------------------------------------------------------
#
# Everything below this line exists because of one incident. `spectra.py` landed with
# the suite above green and was then pointed at its first real case: a cylinder at
# Re = 100 whose `system/controlDict` says `application simpleFoam`, `deltaT 1`,
# `endTime 500`. It reported `shedding: periodic` at 0.0133 Hz, a period of 75, 6.66
# cycles in the record and `St = 0.00133385` -- against the 0.164 that case sheds at,
# which `benchmarks/moving_mesh/grade.py` holds a transient run to within 2% of --
# and `settling: settled -- stationary from index 0` with a mean of 0.00177695 on a
# series whose last value is -0.107. The axis was an iteration count, the peak was in
# bin 3 of 101, and the halves test had been passed by a record that rises and then
# falls. Each test here holds one of those three down.


def write_case_dicts(case: Path, application: str = "simpleFoam", ddt: str = "Euler",
                     delta_t: str = "1", end_time: str = "500") -> Path:
    """The two dictionaries the axis is read from, in OpenFOAM's layout."""
    system = case / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "controlDict").write_text(
        "FoamFile\n{\n    version 2;\n    object controlDict;\n}\n\n"
        f"application     {application};\n\nstartFrom       latestTime;\n\n"
        f"deltaT          {delta_t};\n\nendTime         {end_time};\n\n"
        "writeControl    timeStep;\n\nwriteInterval   1;\n", encoding="utf-8")
    (system / "fvSchemes").write_text(
        "FoamFile\n{\n    version 2;\n    object fvSchemes;\n}\n\n"
        f"ddtSchemes\n{{\n    default         {ddt};\n}}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n", encoding="utf-8")
    return case


def iteration_history(samples: int = 500,
                      period: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """A series on an iteration axis that oscillates hard enough to be called
    `periodic`, so that what the axis suppresses is visible rather than absent."""
    iterations = np.arange(1.0, samples + 1.0)
    return iterations, np.sin(2.0 * np.pi * iterations / period)


def test_a_steady_case_refuses_the_frequency_period_strouhal_and_cycle_count(tmp_path):
    """The incident itself: `application simpleFoam` means the Time column counts
    SIMPLE iterations, and a frequency, a period, a Strouhal number and a count of
    cycles in the record are all meaningless on that axis. The script printed all
    four, to six figures, and the Strouhal it printed was a factor of 100 off the
    0.164 the case sheds at.

    `openreynolds/convergence.py:99-104` already knew which solvers are steady ones
    and nothing here asked it. The series itself is still reported -- the spectrum of
    an iteration history says something about how the solve wandered -- so this
    checks that the four numbers are refused BY NAME and that the record is not
    quietly dropped with them."""
    case = write_case_dicts(tmp_path / "case")
    iterations, values = iteration_history()
    write_dat(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              iterations, {"Cd": np.full(iterations.size, 1.2), "Cl": values})

    found = spectra.analyse(case, column="Cl", length=0.1, speed=1.0)
    assert found["axis"]["kind"] == "iterations"
    assert found["axis"]["assumed"] is False
    assert "simpleFoam" in found["axis"]["basis"]
    assert found["strouhal"] is None, "St = f L / U has no f on an iteration axis"

    text = spectra.report(found)
    assert "time axis: iterations" in text
    for refused in ("frequency", "period", "cycles in record"):
        line = next(row for row in text.splitlines() if row.strip().startswith(refused))
        assert "not reported" in line, line
        assert "iterations" in line, line
    assert "strouhal: not computed" in text
    assert "iterations" in text.split("strouhal: not computed")[1].split("\n")[0]
    assert "Hz" not in text, "no number is in Hz when the axis is not in seconds"
    # Still a report about a series, not a refusal to read one.
    assert f"record: {iterations.size} samples" in text
    assert "tallest bin       bin" in text
    assert "prominence" in text


def test_a_steadystate_ddt_scheme_is_the_second_signal_for_an_iteration_axis(tmp_path):
    """The predicate copied from `convergence.py` is careful about what it claims: a
    `foamRun` with a `steadyState` ddt scheme cannot be told from its name, and it
    does not pretend otherwise. So the axis is read from `system/fvSchemes` as well,
    where `ddtSchemes default steadyState` says it outright -- there is no time
    derivative in the equations being solved, whatever the executable is called. The
    real case carried both signals; a `foamRun` case carries only this one."""
    assert not spectra.is_steady_solver("foamRun"), "the name cannot say, and must not"
    case = write_case_dicts(tmp_path / "case", application="foamRun", ddt="steadyState")
    iterations, values = iteration_history()
    write_dat(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              iterations, {"Cd": np.full(iterations.size, 1.2), "Cl": values})

    found = spectra.analyse(case, column="Cl", length=0.1, speed=1.0)
    assert found["axis"]["kind"] == "iterations"
    assert "fvSchemes" in found["axis"]["basis"]
    assert "steadyState" in found["axis"]["basis"]
    assert found["strouhal"] is None
    assert "not reported" in spectra.report(found)

    # The same dictionaries with a transient solver and a transient scheme read as
    # seconds, or this would pass on a script that has learned to refuse everything.
    moving = write_case_dicts(tmp_path / "transient", application="pimpleFoam", ddt="Euler")
    t, x = sine(1.7, span=20.0, samples=2001)
    write_dat(moving / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cl": x})
    unsteady = spectra.analyse(moving, column="Cl", length=0.1, speed=1.0)
    assert unsteady["axis"]["kind"] == "seconds"
    assert unsteady["axis"]["assumed"] is False
    assert unsteady["strouhal"] is not None


def test_a_bare_dat_with_no_case_says_seconds_are_assumed(tmp_path):
    """Handed a file with no case above it, there is nothing to read the axis from.
    Seconds are then what the numbers are taken to mean, and the report has to say
    that this is an assumption: the frequency it prints is right if the assumption is
    and meaningless if it is not, and the reader is the only one who can tell. Saying
    so is the difference between a reading and a guess."""
    t, x = sine(1.7, span=20.0, samples=2001)
    path = tmp_path / "loose" / "history.csv"
    path.parent.mkdir(parents=True)
    path.write_text("Time,y\n" + "\n".join(f"{a:.8g},{b:.8g}" for a, b in zip(t, x)) + "\n",
                    encoding="utf-8")

    found = spectra.analyse(path, column="y")
    assert found["axis"]["kind"] == "seconds"
    assert found["axis"]["assumed"] is True
    assert "ASSUMED" in found["axis"]["basis"]
    text = spectra.report(found)
    assert "time axis: seconds (ASSUMED)" in text
    # The assumption does not cost the measurement: it is stated, not withheld.
    assert abs(found["peak"]["frequency"] - 1.7) < found["peak"]["resolution"]
    assert "dominant" in text


def test_a_peak_in_the_lowest_bins_is_trend_dominated_and_names_the_bin():
    """`PROMINENCE_PERIODIC`'s own docstring recorded that an AR(1) at 0.98 measures
    2530x and that only the printed frequency would catch it. Nothing caught it: the
    real record measured 2.552e+10x, was called `periodic`, and its peak sat at
    0.0133 Hz on a 0.005 Hz bin grid -- bin 3 of 101, the bottom of the spectrum. A
    median over the whole spectrum is a global floor and cannot tell a slope rising
    towards DC from a peak, so the bin INDEX has to be part of the verdict.

    The fixture is that record's shape: a slow wander of about six cycles per record
    on a decaying transient, which measures bin 3 and 2.5e+10x here too. The cost of
    this rule is stated in `TREND_BINS` and is real -- a genuine tone in the lowest
    three bins is refused the same way, and its route is a longer record, or `--from`
    past the transient."""
    iterations = np.arange(1.0, 501.0)
    values = (0.04 * np.sin(2.0 * np.pi * iterations / 75.0)
              + 0.3 * np.exp(-(iterations - 1.0) / 20.0))
    found = measure(iterations, values)
    assert found["peak"]["bin"] <= spectra.TREND_BINS
    assert found["peak"]["prominence"] > spectra.PROMINENCE_PERIODIC
    assert found["peak"]["periods"] >= spectra.MIN_PERIODS, "not refused for being short"
    assert found["verdict"] == "trend-dominated"
    assert f"bin {found['peak']['bin']}" in found["basis"]
    assert f"of {found['peak']['bins']}" in found["basis"]
    # A tone in the body of the spectrum is still a tone.
    assert measure(*sine(1.7))["verdict"] == "periodic"


def test_a_column_with_no_variation_in_it_is_refused_rather_than_given_a_frequency(tmp_path):
    """Every other test here is a ratio inside the spectrum, and round-off satisfies a
    ratio as well as a signal does: a flat signal does not give a flat spectrum, it
    gives a round-off spectrum whose own median is 1e-67, so peak-over-median
    explodes. 2978 rows of exactly 1.35 behind a real forces header reported
    `St = 0.167926`, inside the published Re = 100 band of 0.164 to 0.167, with a
    prominence of 7.446e+05x.

    Two shapes, two scales, both already set by this codebase against this exact
    failure. The flat column is caught by `FLAT_RELATIVE` against its own magnitude
    (`acoustics.py`). The second is not flat at all -- a real `pressure_z` of a 2D
    extruded case decays smoothly over 6e-6 of itself, a variation the first test
    passes -- and is caught by `NEGLIGIBLE_FRACTION` against the largest value in its
    own record (`report.py:zero_the_negligible`), because the z-force of that case is
    zero by construction and what the solver wrote is the round-off of the summation.
    """
    t = np.arange(1, 2979) * 0.005
    write_dat(tmp_path / "flat" / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cd": np.full(t.size, 1.35), "Cl": np.zeros(t.size)})
    flat = spectra.analyse(tmp_path / "flat", column="Cd", length=0.01, speed=0.01)
    assert flat["peak"]["prominence"] > spectra.PROMINENCE_PERIODIC, "the ratio is fooled"
    assert flat["variation"]["flat"] is True
    assert flat["shedding"]["verdict"] == "no-variation"
    assert flat["strouhal"]["value"] is None, "no St from a column that does not move"
    assert "strouhal: St =" not in spectra.report(flat)

    # Round-off that is not flat: a wander at 1e-30 on a base of 1e-25, beside a
    # sibling at 1e-13. Its RMS about its own trend is 5e-6 of its own magnitude, four
    # million times `FLAT_RELATIVE`, so the first test passes it.
    decay = 5.7346e-25 + 3.0e-30 * np.sin(2.0 * np.pi * 1.7 * t)
    write_dat(tmp_path / "zero" / "postProcessing" / "forces" / "0" / "force.dat",
              t, {"total_y": np.full(t.size, 7.288e-13), "pressure_z": decay})
    zero = spectra.analyse(tmp_path / "zero", column="pressure_z", length=0.05, speed=0.0157)
    assert zero["variation"]["flat"] is False, "it really does vary, 6e-6 of itself"
    assert zero["variation"]["negligible"] is True
    assert zero["shedding"]["verdict"] == "no-variation"
    assert zero["strouhal"]["value"] is None

    # And a real signal at the same tiny absolute level is still a real signal: the
    # scale is the record's own, so nothing here depends on a unit.
    small = 1e-25 * (1.0 + np.sin(2.0 * np.pi * 1.7 * t))
    write_dat(tmp_path / "small" / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cl": small})
    assert spectra.analyse(tmp_path / "small", column="Cl")["shedding"]["verdict"] == "periodic"


def test_no_strouhal_number_is_printed_under_a_verdict_that_rejected_the_peak(tmp_path):
    """A real Cd record spanning 1.70 cycles printed `shedding: too-short` and then
    `strouhal: St = 0.0113328` underneath it with nothing linking the two, and `--json`
    carried them as sibling keys. The same record cropped past its transient gives
    0.332192, so the printed number was 29x low, and a consumer reading
    `strouhal.value` had no way to tell.

    The verdict is computed before the division and is the gate. The quotient is still
    reported, under a name that is not a Strouhal number, because a reader who sees no
    St line concludes the script is broken and does the division themselves.
    """
    t, x = sine(0.01, span=170.0, samples=3401, offset=1.2)
    write_dat(tmp_path / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t, {"Cd": x})
    found = spectra.analyse(tmp_path, column="Cd", length=0.01, speed=0.01)
    assert found["shedding"]["verdict"] == "too-short"
    assert found["strouhal"]["value"] is None
    assert found["strouhal"]["verdict"] == "too-short"
    assert found["strouhal"]["quotient"] == pytest.approx(found["peak"]["frequency"], rel=1e-9)
    text = spectra.report(found)
    assert "strouhal: not reported" in text
    assert "too-short" in text.split("strouhal: not reported")[1]
    assert "strouhal: St =" not in text, "no unqualified St under a rejecting verdict"

    # A verdict that accepts the peak still gets its St, or this would pass on a
    # script that has learned to refuse everything.
    t2, x2 = sine(1.7, span=20.0, samples=2001, offset=1.2)
    write_dat(tmp_path / "long" / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              t2, {"Cd": x2})
    good = spectra.analyse(tmp_path / "long", column="Cd", length=0.01, speed=0.017)
    assert good["shedding"]["verdict"] == "periodic"
    assert good["strouhal"]["value"] == pytest.approx(1.0, rel=0.05)
    assert "strouhal: St =" in spectra.report(good)


def test_the_nyquist_limit_is_reported_and_an_unresolvable_strouhal_is_named(tmp_path):
    """The most useful line the script could have printed about that case, and did
    not. The history was written every iteration of a `deltaT 1` run, so the band it
    resolves tops out at 0.5 per unit, while St = 0.164 at L = 0.1 m and U = 1 m/s
    needs 1.64 per second. Nothing at the shedding frequency could have been in that
    record at any amplitude, so neither the peak that was found nor the absence of
    one said anything at all about the shedding."""
    case = write_case_dicts(tmp_path / "case")
    iterations, values = iteration_history()
    write_dat(case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
              iterations, {"Cl": values})

    found = spectra.analyse(case, column="Cl", length=0.1, speed=1.0, expect=0.164)
    assert found["record"]["nyquist"] == pytest.approx(0.5)
    assert found["requested"]["frequency"] == pytest.approx(1.64)
    assert found["requested"]["resolvable"] is False
    text = spectra.report(found)
    assert "nyquist limit     0.5" in text
    assert "UNRESOLVABLE" in text
    assert "at any amplitude" in text

    # A St the sampling can hold is not called unresolvable.
    inside = spectra.analyse(case, column="Cl", length=0.1, speed=1.0, expect=0.02)
    assert inside["requested"]["resolvable"] is True
    assert "UNRESOLVABLE" not in spectra.report(inside)


# -- settling, after the halves test was found to have a null space ------------


def test_a_record_that_rises_then_falls_is_not_settled_from_index_zero():
    """The third defect, on a fixture built from the real series' shape.

    That Cl read +0.0078 at iteration 1, +0.018 by 250 and -0.107 by 500, with the Cd
    beside it starting at 7.74 and falling to 1.17. Because it is not MONOTONE its
    two halves have nearly the same mean: it scored 0.58 on the halves test, passed
    at sample ZERO, and the mean reported -- 0.00177695 +/- 0.01714 (95%) -- was an
    average over the startup transient of a record whose oscillation had by then
    grown sixfold. `test_a_drifting_series_is_still_moving_and_a_stationary_one_is_settled`
    only ever asked about a ramp, and a ramp, being monotone, is the one shape the
    halves test cannot miss.

    So the fixture is the shape that defeats it: a large decaying transient at the
    start, a mild rise then fall through the middle, and an oscillation that grows
    across the record the way a steady solver losing stability does. What it must not
    report is `settled` from index 0 with a mean carrying the transient."""
    iterations = np.arange(1.0, 501.0)
    transient = 0.08 * np.exp(-(iterations - 1.0) / 10.0)
    rise_then_fall = 0.0078 + 1.2e-4 * (iterations - 1.0) - 4.2e-7 * (iterations - 1.0) ** 2
    growing = (0.010 * np.exp((iterations - 1.0) / 200.0)
               * np.sin(2.0 * np.pi * iterations / 80.0))
    values = transient + rise_then_fall + growing

    found = spectra.settle(iterations, values)
    assert found["verdict"] != "settled"
    assert found["index"] is None, "no window of a record still growing is stationary"
    assert spectra.scatter_growth(values) > spectra.SCATTER_GROWTH
    assert f"{spectra.scatter_growth(values):.2f}" in found["basis"]

    # And the number it does print is the tail's, not the transient's: the first fifty
    # samples average +0.0317 here while the last quarter averages -0.0211, so the two
    # are not even on the same side of zero. Printed, as before, as a number and not a
    # result -- the record is not stationary and the report says which it is.
    assert found["mean"] == pytest.approx(float(np.mean(values[-(values.size // 4):])))
    assert found["mean"] < 0.0 < float(np.mean(values[:50]))


def test_the_halves_test_alone_would_have_passed_that_record():
    """The other half of the same point, and why the two new checks are not redundant
    with the old one. On this fixture the halves ratio is UNDER the threshold -- the
    test the script shipped with saw nothing wrong with the record -- while the
    thirds comparison and the scatter growth refuse it. If a later simplification
    drops either, this fails instead of the record passing again."""
    iterations = np.arange(1.0, 501.0)
    transient = 0.08 * np.exp(-(iterations - 1.0) / 10.0)
    growing = (0.010 * np.exp((iterations - 1.0) / 200.0)
               * np.sin(2.0 * np.pi * iterations / 80.0))
    values = transient + growing

    assert spectra.drift_ratio(values) <= spectra.SETTLED_RATIO, "the null space, measured"
    assert spectra.scatter_growth(values) > spectra.SCATTER_GROWTH
    assert spectra.settle(iterations, values)["verdict"] == "still-moving"

    # A limit cycle on the same axis and of the same amplitude, NOT growing: still
    # settled, or the new checks would have bought a refusal of everything.
    steady = 1.2 + 0.05 * np.sin(2.0 * np.pi * iterations / 80.0)
    assert spectra.settle(iterations, steady)["verdict"] == "settled"


def test_the_density_integrates_to_the_records_mean_square():
    """Parseval, which is the only check that can see a constant factor.

    `welch`'s scale read `1.0 / (dt * sum(w**2))` where a one-sided density needs
    `dt / sum(w**2)`, so every density came back a factor of `1/dt**2` -- the sampling
    rate squared -- too large. A 1 Pa RMS sine sampled at 10 kHz integrated to 1e8
    instead of to 1. Every other test here passed throughout, because a peak's
    frequency, its prominence over the spectrum median and the settling verdict are
    ratios and a constant factor cancels in all of them. It was found when an absolute
    sound pressure level was wanted from the same estimator.
    """
    fs = 10000.0
    t = np.arange(4096) / fs
    x = np.sqrt(2.0) * np.sin(2.0 * np.pi * 200.0 * t)  # 1.0 Pa RMS
    out = spectra.welch(t, x, segments=1)
    integral = float(np.sum(out["power"]) * out["resolution"])
    assert integral == pytest.approx(1.0, rel=0.02), integral
