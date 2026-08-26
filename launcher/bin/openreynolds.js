#!/usr/bin/env node
"use strict";

/**
 * `openreynolds` -- the npm launcher for the openreynolds Python package.
 *
 *     npm install -g openreynolds     # or: npx openreynolds
 *     openreynolds "flow over a cylinder at Re=100"
 *
 * The real program lives on PyPI. This file exists so people who live in Node get a
 * one-command install; it owns exactly three jobs and then gets out of the way:
 *
 *   1. find `uv` (offering the official installer when it is missing),
 *   2. `uv tool install` the Python package once, pinned to PYTHON_PACKAGE_SPEC,
 *   3. hand the terminal to the installed `openreynolds` binary.
 *
 * It is dependency-free on purpose: it is the first thing a new user runs, so it has to
 * work on a bare Node 18 with nothing else present. Every side effect is announced on
 * one line before it happens, and nothing is installed without saying so first.
 */

const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

// Bump this range with every launcher release that tracks a new Python minor. The
// launcher deliberately pins a range rather than an exact version so `openreynolds
// upgrade` can pick up patch releases without an npm update.
const PYTHON_PACKAGE = "openreynolds";
const PYTHON_PACKAGE_SPEC = "openreynolds>=0.1,<0.2";
const UV_DOCS = "https://docs.astral.sh/uv/";
const REPO = "https://github.com/InviscidAI/OpenReynolds";

// -NoProfile: a user's PowerShell profile can fail loudly (OneDrive-redirected
// Documents whose provider is offline) and that noise would look like our failure.
const uvInstallCommand = (platform) =>
  platform === "win32"
    ? 'powershell -NoProfile -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    : "curl -LsSf https://astral.sh/uv/install.sh | sh";

// --- theme --------------------------------------------------------------------------

const DEPTH = { NONE: 0, ANSI16: 1, ANSI256: 2, TRUECOLOR: 3 };
const BRAND = [0x38, 0xbd, 0xf8];
const OK = [0x34, 0xd3, 0x99];
const WARN = [0xfb, 0xbf, 0x24];
const ERR = [0xf8, 0x71, 0x71];
const MUTED = [0x8b, 0x98, 0xa5];
const HEADING = [0xe2, 0xe8, 0xf0];

const UNICODE_GLYPHS = {
  tl: "╭", tr: "╮", bl: "╰", br: "╯", h: "─", v: "│",
  ok: "✓", warn: "▲", bad: "✕", arrow: "→", ellipsis: "…",
};
const ASCII_GLYPHS = {
  tl: "+", tr: "+", bl: "+", br: "+", h: "-", v: "|",
  ok: "+", warn: "!", bad: "x", arrow: "->", ellipsis: "...",
};

const ANSI_RE = /\x1b\[[0-9;]*m/g;
const visibleWidth = (value) => String(value).replace(ANSI_RE, "").length;
const CUBE = [0, 95, 135, 175, 215, 255];
const cubeIndex = (v) =>
  CUBE.reduce((best, level, i) => (Math.abs(level - v) < Math.abs(CUBE[best] - v) ? i : best), 0);

function rgbTo256([r, g, b]) {
  if (Math.abs(r - g) < 12 && Math.abs(g - b) < 12) {
    const gray = Math.round((r + g + b) / 3);
    return gray < 8 ? 16 : gray > 248 ? 231 : 232 + Math.round(((gray - 8) / 247) * 23);
  }
  return 16 + 36 * cubeIndex(r) + 6 * cubeIndex(g) + cubeIndex(b);
}

function rgbTo16([r, g, b]) {
  const base = (r > 110 ? 1 : 0) + (g > 110 ? 2 : 0) + (b > 110 ? 4 : 0);
  return (Math.max(r, g, b) > 170 ? 90 : 30) + base;
}

function detectDepth(env, isTty) {
  if (env.NO_COLOR) return DEPTH.NONE;
  if (!isTty && !env.FORCE_COLOR) return DEPTH.NONE;
  const term = (env.TERM || "").toLowerCase();
  if (term === "dumb") return DEPTH.NONE;
  const colorterm = (env.COLORTERM || "").toLowerCase();
  if (colorterm.includes("truecolor") || colorterm.includes("24bit") || env.WT_SESSION) return DEPTH.TRUECOLOR;
  const program = (env.TERM_PROGRAM || "").toLowerCase();
  if (["vscode", "iterm.app", "wezterm", "ghostty", "hyper", "tabby"].includes(program)) return DEPTH.TRUECOLOR;
  return term ? DEPTH.ANSI256 : DEPTH.ANSI16;
}

// Node does not expose the console code page, so on Windows we need positive evidence
// of a modern terminal before emitting anything outside ASCII. OPENREYNOLDS_ASCII=1
// forces the fallback everywhere.
function detectUnicode(env, stream, platform) {
  if (env.OPENREYNOLDS_ASCII) return false;
  const encoding = String(stream.encoding || "").toLowerCase();
  if (encoding && !/utf-?8/.test(encoding)) return false;
  if (platform !== "win32") return true;
  return Boolean(env.WT_SESSION || env.TERM_PROGRAM || env.MSYSTEM || env.TERM || env.ConEmuANSI);
}

function makeTheme(stream, env, platform) {
  const isTty = Boolean(stream.isTTY);
  const depth = detectDepth(env, isTty);
  const glyphs = detectUnicode(env, stream, platform) ? UNICODE_GLYPHS : ASCII_GLYPHS;
  // Piped output has no column limit; the install command must never be clipped there.
  const width = isTty ? Math.max(52, Math.min(104, (stream.columns || 88) - 1)) : 120;
  const fg = (value, color) => {
    if (depth === DEPTH.NONE || !value) return value;
    const code =
      depth === DEPTH.TRUECOLOR ? `38;2;${color.join(";")}`
      : depth === DEPTH.ANSI256 ? `38;5;${rgbTo256(color)}`
      : String(rgbTo16(color));
    return `\x1b[${code}m${value}\x1b[39m`;
  };
  const sgr = (value, code) => (depth === DEPTH.NONE || !value ? value : `\x1b[${code}m${value}\x1b[0m`);
  return {
    glyphs, width, isTty, fg,
    bold: (v) => sgr(v, "1"),
    brand: (v) => fg(v, BRAND),
    good: (v) => fg(v, OK),
    warning: (v) => fg(v, WARN),
    bad: (v) => fg(v, ERR),
    heading: (v) => sgr(fg(v, HEADING), "1"),
    muted: (v) => sgr(fg(v, MUTED), "2"),
  };
}

function panel(theme, rows, { title = "", color = BRAND } = {}) {
  const g = theme.glyphs;
  const inner = Math.min(Math.max(...rows.map(visibleWidth), visibleWidth(title) + 4, 24), theme.width - 4);
  const side = theme.fg(g.v, color);
  const head = title ? ` ${theme.bold(theme.fg(title, color))} ` : "";
  const bar = g.h.repeat(Math.max(0, inner + 2 - visibleWidth(head) - (title ? 1 : 0)));
  const top = theme.fg(g.tl + (title ? g.h : ""), color) + head + theme.fg(bar + g.tr, color);
  const bottom = theme.fg(g.bl + g.h.repeat(inner + 2) + g.br, color);
  const clip = (row) =>
    visibleWidth(row) <= inner ? row : String(row).replace(ANSI_RE, "").slice(0, inner - g.ellipsis.length) + g.ellipsis;
  const body = rows.map((row) => `${side} ${clip(row)}${" ".repeat(Math.max(0, inner - visibleWidth(clip(row))))} ${side}`);
  return [top, ...body, bottom].join("\n");
}

// --- process helpers ----------------------------------------------------------------

// Quote one argv entry for cmd.exe; Node does not quote for you when `shell: true`.
const winQuote = (value) =>
  /[\s"&|<>^()%!]/.test(value) ? `"${String(value).replace(/"/g, '""')}"` : String(value);

// A freshly installed uv.exe has reported `spawn UNKNOWN` on a clean Windows machine:
// CreateProcess refused the image directly while launching through cmd.exe worked.
// ENOENT/EACCES also cover .cmd/.bat shims.
const RETRYABLE_SPAWN = new Set(["UNKNOWN", "ENOENT", "EACCES", "EINVAL"]);
const shouldRetryInShell = (error, platform) =>
  platform === "win32" && Boolean(error) && RETRYABLE_SPAWN.has(error.code);

const sleepSync = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);

/** Run to completion, capturing output. Never throws. */
function runCapture(command, args, { env = process.env, timeout } = {}) {
  const once = (cmd, argv, shell) => {
    const r = spawnSync(cmd, argv, { env, shell, timeout, windowsHide: true, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    return { status: r.status, stdout: r.stdout || "", stderr: r.stderr || "", error: r.error || null };
  };
  const first = once(command, args, false);
  if (!shouldRetryInShell(first.error, process.platform)) return first;
  return once(winQuote(command), args.map(winQuote), true);
}

/** Run with the terminal attached (installers). Never throws. */
function runInherit(command, args, { env = process.env } = {}) {
  const once = (cmd, argv, shell) => spawnSync(cmd, argv, { env, shell, stdio: "inherit" });
  const first = once(command, args, false);
  if (!shouldRetryInShell(first.error, process.platform)) return first;
  return once(winQuote(command), args.map(winQuote), true);
}

/**
 * Hand the terminal to the real program and resolve with its exit code.
 *
 * Ctrl-C is delivered by the terminal to the whole foreground process group, so the
 * child already receives it; the launcher must neither die first (which would orphan
 * the Python process mid-cleanup) nor forward a second SIGINT on top. SIGTERM/SIGHUP
 * target one pid, so those are relayed.
 */
function execInherit(command, args, { env = process.env } = {}) {
  return new Promise((resolve) => {
    const start = (cmd, argv, shell) => {
      const child = spawn(cmd, argv, { env, shell, stdio: "inherit" });
      const relay = (signal) => () => { try { child.kill(signal); } catch { /* already gone */ } };
      const swallow = () => {};
      const handlers = [["SIGINT", swallow], ["SIGTERM", relay("SIGTERM")], ["SIGHUP", relay("SIGHUP")]];
      for (const [signal, handler] of handlers) process.on(signal, handler);
      const done = (code) => {
        for (const [signal, handler] of handlers) process.removeListener(signal, handler);
        resolve(code);
      };
      child.on("error", (error) => {
        if (!shell && shouldRetryInShell(error, process.platform)) {
          for (const [signal, handler] of handlers) process.removeListener(signal, handler);
          start(winQuote(cmd), argv.map(winQuote), true);
          return;
        }
        process.stderr.write(`openreynolds: could not start ${cmd}: ${error.message}\n`);
        done(1);
      });
      child.on("exit", (code, signal) => {
        if (signal) done(128 + ((os.constants.signals || {})[signal] || 2));
        else done(code === null ? 1 : code);
      });
    };
    start(command, args, false);
  });
}

// --- prompts ------------------------------------------------------------------------

class AbortError extends Error {}

/** Blocking y/N prompt. Only ever used when both ends are a real terminal. */
function confirmSync(theme, question) {
  process.stdout.write(`  ${theme.warning("?")} ${question} ${theme.muted("[y/N]")} `);
  const buffer = Buffer.alloc(1);
  let value = "";
  for (let guard = 0; guard < 60000; guard += 1) {
    let n = 0;
    try { n = fs.readSync(0, buffer, 0, 1, null); }
    catch (error) {
      if (error.code === "EAGAIN") { sleepSync(10); continue; }
      if (error.code === "EOF") break;
      throw error;
    }
    if (n === 0) break;
    const ch = buffer.toString("utf8");
    if (ch === "\n" || ch === "\x04") break;
    if (ch === "\x03") throw new AbortError();
    if (ch !== "\r") value += ch;
  }
  return /^(y|yes)$/i.test(value.trim());
}

// --- injectable environment ---------------------------------------------------------

/** Everything with a side effect goes through here so the tests can script it. */
function defaultDeps() {
  return {
    env: process.env,
    platform: process.platform,
    home: os.homedir(),
    stdout: process.stdout,
    isFile: (p) => { try { return fs.statSync(p).isFile(); } catch { return false; } },
    isDir: (p) => { try { return fs.statSync(p).isDirectory(); } catch { return false; } },
    interactive: () => Boolean(process.stdin.isTTY && process.stdout.isTTY),
    confirm: confirmSync,
    runCapture,
    runInherit,
    exec: execInherit,
    launcherVersion: () => {
      try { return require(path.join(__dirname, "..", "package.json")).version; } catch { return "0.0.0"; }
    },
  };
}

const say = (deps, line = "") => deps.stdout.write(`${line}\n`);
// Path rules follow the platform being *described*, so the tests can script win32 on a mac.
const pathFor = (deps) => (deps.platform === "win32" ? path.win32 : path.posix);

// --- uv -----------------------------------------------------------------------------

function findOnPath(name, deps) {
  const exts = deps.platform === "win32" ? (deps.env.PATHEXT || ".EXE;.CMD;.BAT;.COM").split(";") : [""];
  const P = pathFor(deps);
  for (const raw of (deps.env.PATH || "").split(P.delimiter)) {
    const dir = raw.replace(/^"|"$/g, "").trim();
    if (!dir) continue;
    for (const ext of exts) {
      const candidate = P.join(dir, name + ext);
      if (deps.isFile(candidate)) return candidate;
    }
  }
  return null;
}

// Where the official installer (and Homebrew/winget) put uv when it is not on PATH yet.
function uvInstallDirs(deps) {
  const P = pathFor(deps);
  const dirs = [P.join(deps.home, ".local", "bin"), P.join(deps.home, ".cargo", "bin")];
  if (deps.platform === "win32") {
    const local = deps.env.LOCALAPPDATA || P.join(deps.home, "AppData", "Local");
    dirs.push(P.join(local, "Microsoft", "WinGet", "Links"));
  } else {
    dirs.push("/opt/homebrew/bin", "/usr/local/bin");
  }
  return dirs;
}

function findUv(deps) {
  const explicit = deps.env.OPENREYNOLDS_UV;
  if (explicit) {
    if (deps.isFile(explicit)) return explicit;
    say(deps, `  openreynolds: OPENREYNOLDS_UV=${explicit} is not a file; ignoring it.`);
  }
  const onPath = findOnPath("uv", deps);
  if (onPath) return onPath;
  const names = deps.platform === "win32" ? ["uv.exe", "uv"] : ["uv"];
  for (const dir of uvInstallDirs(deps)) {
    for (const name of names) {
      const candidate = pathFor(deps).join(dir, name);
      if (deps.isFile(candidate)) return candidate;
    }
  }
  return null;
}

function uvMissingPanel(theme, platform) {
  return panel(theme, [
    theme.heading("uv is required and was not found."),
    "",
    "openreynolds is a Python program; uv installs it (and a Python 3.10+ if needed).",
    "Install uv with:",
    theme.brand(uvInstallCommand(platform)),
    "",
    theme.muted(`Then re-run \`openreynolds\`. Non-interactive? Pass --yes or set OPENREYNOLDS_YES=1`),
    theme.muted(`to let the launcher run that installer for you. Docs: ${UV_DOCS}`),
  ], { title: "missing dependency", color: ERR });
}

function installUv(theme, deps) {
  say(deps, `  ${theme.glyphs.arrow} Installing uv with the official installer${theme.glyphs.ellipsis}`);
  say(deps, `  ${theme.muted(`$ ${uvInstallCommand(deps.platform)}`)}`);
  say(deps);
  const result = deps.platform === "win32"
    ? deps.runInherit("powershell", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-c", "irm https://astral.sh/uv/install.ps1 | iex"])
    : deps.runInherit("/bin/sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]);
  say(deps);
  if (result.error || result.status !== 0) return null;
  // The installer edits the *persistent* PATH; this process still has the old one.
  const { delimiter } = pathFor(deps);
  for (const dir of uvInstallDirs(deps).slice(0, 2).reverse()) {
    if (!(deps.env.PATH || "").split(delimiter).includes(dir)) deps.env.PATH = `${dir}${delimiter}${deps.env.PATH || ""}`;
  }
  return findUv(deps);
}

/** Returns the uv path, or an exit code when the launcher cannot go on. */
function ensureUv(theme, deps, { yes }) {
  let uv = findUv(deps);
  if (uv) return { uv };
  const g = theme.glyphs;
  say(deps, `  ${theme.bad(g.bad)} uv not found ${theme.muted("(PATH, ~/.local/bin, ~/.cargo/bin)")}`);
  let agreed = yes;
  if (!agreed && deps.interactive()) {
    say(deps);
    say(deps, `  ${theme.heading("openreynolds needs uv to install and run its Python package.")}`);
    say(deps, `  ${theme.muted(uvInstallCommand(deps.platform))}`);
    say(deps);
    agreed = deps.confirm(theme, "Run the official uv installer now?");
  }
  if (!agreed) {
    say(deps);
    say(deps, uvMissingPanel(theme, deps.platform));
    return { exit: 2 };
  }
  uv = installUv(theme, deps);
  if (!uv) {
    say(deps, `  ${theme.bad(g.bad)} the installer did not leave a usable uv on this machine`);
    say(deps, uvMissingPanel(theme, deps.platform));
    return { exit: 1 };
  }
  // A binary written seconds ago can be briefly unlaunchable while antivirus scans it
  // or a cloud-sync filter materializes it; a short retry turns that into a non-event.
  let probe = deps.runCapture(uv, ["--version"], { env: deps.env, timeout: 15000 });
  for (let attempt = 1; attempt < 3 && (probe.error || probe.status !== 0); attempt += 1) {
    sleepSync(400 * attempt);
    probe = deps.runCapture(uv, ["--version"], { env: deps.env, timeout: 15000 });
  }
  if (probe.error || probe.status !== 0) {
    say(deps, panel(theme, [
      theme.heading("uv is installed but this process cannot start it."),
      theme.muted(`  path:  ${uv}`),
      theme.muted(`  error: ${probe.error ? probe.error.code || probe.error.message : `exit ${probe.status}`}`),
      "",
      theme.muted("Open a NEW terminal and run openreynolds again -- the installer edits PATH and a"),
      theme.muted("brand-new binary can stay locked briefly while antivirus scans it. If that fails,"),
      theme.muted("verify by hand:"),
      theme.brand(`  "${uv}" --version`),
    ], { title: "uv not runnable", color: ERR }));
    return { exit: 1 };
  }
  say(deps, `  ${theme.good(g.ok)} uv ${theme.muted(`${probe.stdout.trim()}  ${uv}`)}`);
  return { uv };
}

// --- Python package -----------------------------------------------------------------

/** `uv tool list` prints one `name vX.Y.Z` header per tool, entry points indented below. */
function isToolInstalled(listOutput, name = PYTHON_PACKAGE) {
  const pattern = new RegExp(`^${name.replace(/[-_]/g, "[-_]")}\\s+v\\S+`, "im");
  return pattern.test(String(listOutput || ""));
}

function resolveSpec(env) {
  const override = String(env.OPENREYNOLDS_PYTHON_SPEC || "").trim();
  return override || PYTHON_PACKAGE_SPEC;
}

/** Install the package if `uv tool list` does not show it. Returns an exit code or 0. */
function ensurePackage(theme, deps, uv, { spec, fromPath, upgrade }) {
  const g = theme.glyphs;
  let args;
  if (fromPath) {
    const dir = path.resolve(fromPath);
    if (!deps.isDir(dir)) {
      say(deps, `  ${theme.bad(g.bad)} --from-path ${dir} is not a directory`);
      return 1;
    }
    say(deps, `  ${g.arrow} Installing the openreynolds Python package from ${dir} (editable) with uv${g.ellipsis}`);
    args = ["tool", "install", "--editable", dir];
  } else {
    const listed = deps.runCapture(uv, ["tool", "list"], { env: deps.env, timeout: 30000 });
    const installed = isToolInstalled(`${listed.stdout}\n${listed.stderr}`);
    if (installed && !upgrade) return 0;
    // `uv tool upgrade` keeps the requirement recorded at install time, so a launcher
    // whose pin moved to a new minor would never get there; re-installing with the
    // current spec and --upgrade covers both "newer patch" and "new range".
    say(deps, upgrade
      ? `  ${g.arrow} Upgrading the openreynolds Python package with uv (${spec})${g.ellipsis}`
      : `  ${g.arrow} Installing the openreynolds Python package with uv (one time)${g.ellipsis}`);
    args = upgrade ? ["tool", "install", "--upgrade", spec] : ["tool", "install", spec];
  }
  say(deps, `  ${theme.muted(`$ uv ${args.map(winQuote).join(" ")}`)}`);
  const result = deps.runInherit(uv, args, { env: deps.env });
  if (result.error || result.status !== 0) {
    say(deps);
    say(deps, panel(theme, [
      theme.heading("uv could not install the openreynolds Python package."),
      theme.muted(result.error ? result.error.message : `uv exited with ${result.status}`),
      "",
      "Try it by hand to see the full output:",
      theme.brand(`  uv ${args.map(winQuote).join(" ")}`),
      theme.muted("A Python 3.10+ is fetched by uv automatically; `uv python install 3.12` forces one."),
      theme.muted(`Still stuck? ${REPO}/issues`),
    ], { title: "install failed", color: ERR }));
    return 1;
  }
  say(deps, `  ${theme.good(g.ok)} ${upgrade ? "upgraded" : "installed"} ${theme.muted(`(uv tool ${PYTHON_PACKAGE})`)}`);
  return 0;
}

/** The binary uv just installed. `uv tool dir --bin` honors UV_TOOL_BIN_DIR for us. */
function findToolBinary(deps, uv) {
  const P = pathFor(deps);
  const name = deps.platform === "win32" ? `${PYTHON_PACKAGE}.exe` : PYTHON_PACKAGE;
  const dirs = [];
  const asked = deps.runCapture(uv, ["tool", "dir", "--bin"], { env: deps.env, timeout: 15000 });
  if (asked.status === 0 && asked.stdout.trim()) dirs.push(asked.stdout.trim());
  dirs.push(P.join(deps.home, ".local", "bin"));
  for (const dir of dirs) {
    const candidate = P.join(dir, name);
    if (deps.isFile(candidate)) return candidate;
  }
  return null;
}

// --- entry point --------------------------------------------------------------------

/** Launcher-owned argv; everything else is forwarded verbatim to the Python CLI. */
function parseArgs(argv) {
  const out = { yes: false, fromPath: null, launcherVersion: false, upgrade: false, passthrough: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--launcher-version") out.launcherVersion = true;
    else if (arg === "--yes" && out.passthrough.length === 0) out.yes = true;
    else if (arg === "--from-path" && i + 1 < argv.length) out.fromPath = argv[++i];
    else if (arg.startsWith("--from-path=")) out.fromPath = arg.slice("--from-path=".length);
    else if (arg === "upgrade" && out.passthrough.length === 0) out.upgrade = true;
    else out.passthrough.push(arg);
  }
  return out;
}

async function main(argv, deps = defaultDeps()) {
  const env = argv.includes("--no-color") ? { ...deps.env, NO_COLOR: "1" } : deps.env;
  const theme = makeTheme(deps.stdout, env, deps.platform);
  const args = parseArgs(argv);

  if (args.launcherVersion) {
    say(deps, `${theme.brand("openreynolds")} launcher ${deps.launcherVersion()} ${theme.muted(`(python spec ${resolveSpec(deps.env)}; node ${process.versions.node} ${deps.platform}-${process.arch})`)}`);
    return 0;
  }

  const yes = args.yes || /^(1|true|yes)$/i.test(String(deps.env.OPENREYNOLDS_YES || ""));
  const found = ensureUv(theme, deps, { yes });
  if (!found.uv) return found.exit;

  const installed = ensurePackage(theme, deps, found.uv, {
    spec: resolveSpec(deps.env), fromPath: args.fromPath, upgrade: args.upgrade,
  });
  if (installed !== 0) return installed;
  if (args.upgrade) return 0;

  const binary = findToolBinary(deps, found.uv);
  if (!binary) {
    say(deps, panel(theme, [
      theme.heading("The openreynolds Python package is installed but its binary was not found."),
      theme.muted("Ask uv where it puts tool binaries and check that directory:"),
      theme.brand("  uv tool dir --bin"),
      theme.muted(`Report it at ${REPO}/issues if it is really missing.`),
    ], { title: "cannot start", color: ERR }));
    return 1;
  }
  // Prepend uv's tool bin dir so the child's own shell-outs resolve even when the
  // user's PATH never picked up uv's edits.
  const P = pathFor(deps);
  const childEnv = { ...deps.env, PATH: `${P.dirname(binary)}${P.delimiter}${deps.env.PATH || ""}` };
  return deps.exec(binary, args.passthrough, { env: childEnv });
}

module.exports = {
  PYTHON_PACKAGE, PYTHON_PACKAGE_SPEC, uvInstallCommand, AbortError,
  makeTheme, panel, parseArgs, findOnPath, findUv, uvInstallDirs, isToolInstalled,
  resolveSpec, ensureUv, ensurePackage, findToolBinary, defaultDeps, main,
};

if (require.main === module) {
  process.stdout.on("error", (error) => { if (error && error.code === "EPIPE") process.exit(0); });
  main(process.argv.slice(2)).then(
    (status) => process.exit(status),
    (error) => {
      if (error instanceof AbortError) { process.stdout.write("\n"); process.exit(130); }
      const theme = makeTheme(process.stdout, process.env, process.platform);
      process.stdout.write(`\n${panel(theme, [theme.heading("openreynolds launcher failed."), theme.muted(String(error && error.message ? error.message : error))], { title: "unexpected error", color: ERR })}\n`);
      process.exit(1);
    },
  );
}
