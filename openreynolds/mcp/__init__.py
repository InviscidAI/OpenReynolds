"""Reynolds as an MCP server: another agent hands a CFD task to this one.

The caller is a coding agent -- Claude Code, or any MCP client -- and what it wants is
what a person at a Reynolds session wants: say what to simulate, hand over a geometry,
watch, and get the answer and the files back. So the tools are a session's own verbs.

There are two servers and one set of tools. `core.build_server` defines the tools --
their names, arguments, descriptions and answers, and how a study's state is read --
against a `core.Host`, which is everything that differs between where a study runs:

* `local.LocalHost` (`openreynolds mcp`): studies run in this process, on this
  machine's workspace or a hosted one, exactly as `openreynolds` runs them in a
  terminal. Served over stdio.
* the hosted Reynolds service implements the same `Host` over its own sessions, and
  serves the same tools over HTTP.

A caller that learns one learns both.
"""
