"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const launcher = require("../bin/openreynolds.js");

const TOOL_LIST_INSTALLED = "openreynolds v0.1.2\n- openreynolds\n";
const TOOL_LIST_OTHER = "ruff v0.6.0\n- ruff\n";
const BIN_DIR = "/home/u/.local/bin";

/**
 * A scripted environment. Every side effect the launcher can have is recorded in
 * `calls`; `files` is the set of paths that "exist"; `toolList` is what `uv tool list`
 * prints; `exitCode` is what the final exec returns.
 */
function fakeDeps(overrides = {}) {
  const out = [];
  const calls = { capture: [], inherit: [], exec: [], confirm: 0 };
  // Compare slash-normalized so the win32 scenarios pass on a POSIX host and vice versa.
  const norm = (p) => String(p).replace(/\\/g, "/");
  const files = new Set((overrides.files || []).map(norm));
  const dirs = new Set((overrides.dirs || []).map(norm));
  // A tool that `uv tool list` reports also has its binary on disk, like the real thing.
  if (/^openreynolds v/m.test(overrides.toolList || "")) files.add(`${BIN_DIR}/openreynolds`);
  const deps = {
    env: { PATH: "/usr/bin", ...(overrides.env || {}) },
    platform: overrides.platform || "linux",
    home: "/home/u",
    stdout: { isTTY: false, write: (s) => out.push(s) },
    isFile: (p) => files.has(norm(p)),
    isDir: (p) => dirs.has(norm(p)),
    interactive: () => Boolean(overrides.interactive),
    confirm: () => { calls.confirm += 1; return Boolean(overrides.answer); },
    runCapture: (cmd, args, opts) => {
      calls.capture.push([cmd, ...args]);
      const verb = args.join(" ");
      if (verb === "tool list") return { status: 0, stdout: overrides.toolList || "", stderr: "", error: null };
      if (verb === "tool dir --bin") return { status: 0, stdout: `${BIN_DIR}\n`, stderr: "", error: null };
      if (verb === "--version") return { status: 0, stdout: "uv 0.11.0\n", stderr: "", error: null };
      return { status: 1, stdout: "", stderr: `unexpected: ${verb}`, error: null };
    },
    runInherit: (cmd, args, opts) => {
      calls.inherit.push([cmd, ...args]);
      // Installing the tool makes its binary appear, like the real thing.
      if (args[0] === "tool" && args[1] === "install") files.add(`${BIN_DIR}/openreynolds`);
      if (overrides.installerCreatesUv) files.add("/home/u/.local/bin/uv");
      return { status: overrides.installStatus === undefined ? 0 : overrides.installStatus, error: null };
    },
    exec: async (cmd, args, opts) => {
      calls.exec.push({ cmd, args, env: opts.env });
      return overrides.exitCode === undefined ? 0 : overrides.exitCode;
    },
    launcherVersion: () => "9.9.9",
  };
  return { deps, calls, out, text: () => out.join("") };
}

const withUv = (extra = {}) =>
  fakeDeps({ files: ["/usr/bin/uv", ...(extra.files || [])], toolList: TOOL_LIST_INSTALLED, ...extra });

test("findUv: discovers uv from a fake PATH, honoring PATHEXT on Windows", () => {
  const posix = fakeDeps({ files: ["/opt/tools/uv"], env: { PATH: "/usr/bin:/opt/tools" } });
  assert.equal(launcher.findUv(posix.deps), "/opt/tools/uv");

  const win = fakeDeps({
    platform: "win32",
    files: ["C:\\tools\\uv.EXE"],
    env: { PATH: '"C:\\tools";C:\\other', PATHEXT: ".COM;.EXE" },
  });
  assert.equal(launcher.findUv(win.deps), "C:\\tools\\uv.EXE");
});

test("findUv: OPENREYNOLDS_UV wins over PATH; falls back to ~/.local/bin; null when absent", () => {
  const explicit = fakeDeps({ files: ["/custom/uv", "/usr/bin/uv"], env: { OPENREYNOLDS_UV: "/custom/uv" } });
  assert.equal(launcher.findUv(explicit.deps), "/custom/uv");

  const fallback = fakeDeps({ files: ["/home/u/.local/bin/uv"] });
  assert.equal(launcher.findUv(fallback.deps), "/home/u/.local/bin/uv");

  assert.equal(launcher.findUv(fakeDeps().deps), null);
});

test("isToolInstalled: reads the `uv tool list` header line only", () => {
  assert.equal(launcher.isToolInstalled(TOOL_LIST_INSTALLED), true);
  assert.equal(launcher.isToolInstalled(`${TOOL_LIST_OTHER}${TOOL_LIST_INSTALLED}`), true);
  assert.equal(launcher.isToolInstalled("openreynolds v0.1.0 (from file:///src/OpenReynolds)\n- openreynolds\n"), true);
  assert.equal(launcher.isToolInstalled(TOOL_LIST_OTHER), false);
  assert.equal(launcher.isToolInstalled("No tools installed\n"), false);
  // An entry point line alone (e.g. under another tool) is not an install.
  assert.equal(launcher.isToolInstalled("other v1.0.0\n- openreynolds\n"), false);
});

test("already installed: no install call, exec runs with pass-through argv", async () => {
  const f = withUv();
  const code = await launcher.main(["--verbose", "run", "--model", "x y"], f.deps);
  assert.equal(code, 0);
  assert.equal(f.calls.inherit.length, 0, "must not install again");
  assert.deepEqual(f.calls.capture.map((c) => c.slice(1).join(" ")), ["tool list", "tool dir --bin"]);
  assert.equal(f.calls.exec.length, 1);
  assert.equal(f.calls.exec[0].cmd, `${BIN_DIR}/openreynolds`);
  assert.deepEqual(f.calls.exec[0].args, ["--verbose", "run", "--model", "x y"]);
  assert.ok(f.calls.exec[0].env.PATH.startsWith(`${BIN_DIR}:`), "tool bin dir is prepended to the child's PATH");
});

test("not installed: announces then runs `uv tool install <pinned spec>` before exec", async () => {
  const f = withUv({ toolList: TOOL_LIST_OTHER });
  const code = await launcher.main(["doctor"], f.deps);
  assert.equal(code, 0);
  assert.deepEqual(f.calls.inherit, [["/usr/bin/uv", "tool", "install", launcher.PYTHON_PACKAGE_SPEC]]);
  assert.match(f.text(), /Installing the openreynolds Python package with uv \(one time\)/);
  assert.deepEqual(f.calls.exec[0].args, ["doctor"]);
});

test("spec override: OPENREYNOLDS_PYTHON_SPEC replaces the pinned range", async () => {
  const f = withUv({ toolList: "", env: { OPENREYNOLDS_PYTHON_SPEC: "openreynolds==0.1.7" } });
  await launcher.main([], f.deps);
  assert.deepEqual(f.calls.inherit[0].slice(1), ["tool", "install", "openreynolds==0.1.7"]);
  assert.equal(launcher.resolveSpec({}), launcher.PYTHON_PACKAGE_SPEC);
});

test("install failure: reports and returns 1 without exec", async () => {
  const f = withUv({ toolList: "", installStatus: 1 });
  const code = await launcher.main([], f.deps);
  assert.equal(code, 1);
  assert.equal(f.calls.exec.length, 0);
  assert.match(f.text(), /could not install/);
});

test("exit code and launcher flags: child status is propagated, flags are stripped", async () => {
  const f = withUv({ exitCode: 42 });
  const code = await launcher.main(["--yes", "studies", "--yes"], f.deps);
  assert.equal(code, 42);
  // A leading --yes is the launcher's; a later one belongs to the Python CLI.
  assert.deepEqual(f.calls.exec[0].args, ["studies", "--yes"]);
});

test("upgrade verb: re-installs with --upgrade under the current spec and does not exec", async () => {
  const f = withUv();
  const code = await launcher.main(["upgrade"], f.deps);
  assert.equal(code, 0);
  assert.deepEqual(f.calls.inherit, [["/usr/bin/uv", "tool", "install", "--upgrade", launcher.PYTHON_PACKAGE_SPEC]]);
  assert.equal(f.calls.exec.length, 0);
  assert.match(f.text(), /Upgrading the openreynolds Python package/);
});

test("--from-path: editable install of a local checkout, always, then exec", async () => {
  const dir = path.resolve("/src/OpenReynolds");
  const f = withUv({ dirs: [dir] });
  const code = await launcher.main(["--from-path", "/src/OpenReynolds", "doctor"], f.deps);
  assert.equal(code, 0);
  assert.deepEqual(f.calls.inherit, [["/usr/bin/uv", "tool", "install", "--editable", dir]]);
  assert.ok(!f.calls.capture.some((c) => c.slice(1).join(" ") === "tool list"), "skips the installed check");
  assert.deepEqual(f.calls.exec[0].args, ["doctor"]);

  const bad = withUv();
  assert.equal(await launcher.main(["--from-path=/nope"], bad.deps), 1);
});

test("uv missing, non-interactive, no --yes: prints instructions and exits 2", async () => {
  const f = fakeDeps();
  const code = await launcher.main(["run"], f.deps);
  assert.equal(code, 2);
  assert.equal(f.calls.inherit.length, 0, "installer must not run");
  assert.equal(f.calls.exec.length, 0);
  assert.equal(f.calls.confirm, 0);
  assert.match(f.text(), /uv is required and was not found/);
  assert.match(f.text(), /curl -LsSf https:\/\/astral\.sh\/uv\/install\.sh \| sh/);
  assert.match(f.text(), /--yes/);
});

test("uv missing with --yes / OPENREYNOLDS_YES: announces and runs the official installer, then continues", async () => {
  for (const setup of [{ argv: ["--yes"], env: {} }, { argv: [], env: { OPENREYNOLDS_YES: "1" } }]) {
    const f = fakeDeps({ env: setup.env, installerCreatesUv: true, toolList: TOOL_LIST_INSTALLED });
    const code = await launcher.main(setup.argv, f.deps);
    assert.equal(code, 0);
    assert.deepEqual(f.calls.inherit[0], ["/bin/sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]);
    assert.match(f.text(), /Installing uv with the official installer/);
    assert.equal(f.calls.exec[0].cmd, `${BIN_DIR}/openreynolds`);
    assert.equal(f.calls.confirm, 0);
    assert.ok(f.deps.env.PATH.startsWith("/home/u/.local/bin:"), "install dir is added to PATH");
  }
});

test("uv missing on a TTY: asks first; declining exits 2 without installing", async () => {
  const declined = fakeDeps({ interactive: true, answer: false });
  assert.equal(await launcher.main([], declined.deps), 2);
  assert.equal(declined.calls.confirm, 1);
  assert.equal(declined.calls.inherit.length, 0);

  const accepted = fakeDeps({ interactive: true, answer: true, installerCreatesUv: true, toolList: TOOL_LIST_INSTALLED });
  assert.equal(await launcher.main([], accepted.deps), 0);
  assert.equal(accepted.calls.inherit[0][0], "/bin/sh");
});

test("Windows: installer goes through powershell -NoProfile", async () => {
  const f = fakeDeps({ platform: "win32", env: { PATH: "C:\\Windows" }, installerCreatesUv: false });
  await launcher.main(["--yes"], f.deps);
  assert.equal(f.calls.inherit[0][0], "powershell");
  assert.ok(f.calls.inherit[0].includes("-NoProfile"));
  assert.ok(f.calls.inherit[0].some((a) => a.includes("astral.sh/uv/install.ps1")));
});

test("--launcher-version prints the npm version and pinned spec without touching uv", async () => {
  const f = fakeDeps();
  assert.equal(await launcher.main(["--launcher-version"], f.deps), 0);
  assert.match(f.text(), /openreynolds launcher 9\.9\.9/);
  assert.match(f.text(), new RegExp(launcher.PYTHON_PACKAGE_SPEC.replace(/[.<>=]/g, "\\$&")));
  assert.equal(f.calls.capture.length, 0);
});

test("theme: OPENREYNOLDS_ASCII forces ASCII glyphs; NO_COLOR strips escapes", () => {
  const ascii = launcher.makeTheme({ isTTY: true, columns: 80 }, { OPENREYNOLDS_ASCII: "1", NO_COLOR: "1" }, "linux");
  assert.equal(ascii.glyphs.ok, "+");
  assert.equal(ascii.brand("x"), "x");
  const unicode = launcher.makeTheme({ isTTY: true, columns: 80 }, { COLORTERM: "truecolor" }, "linux");
  assert.equal(unicode.glyphs.ok, "✓");
  assert.match(unicode.brand("x"), /\x1b\[38;2;56;189;248mx\x1b\[39m/);
  // Windows without a modern-terminal marker must stay ASCII.
  assert.equal(launcher.makeTheme({ isTTY: true }, {}, "win32").glyphs.ok, "+");
  assert.equal(launcher.makeTheme({ isTTY: true }, { WT_SESSION: "1" }, "win32").glyphs.ok, "✓");
});
