# Changing the model or effort mid-study

A study can outlive the model you started it on. You may want Opus for the hard
geometry and Sonnet for a long solve that mostly waits, or a different provider when a
budget runs out. `/model` and `/effort` change either without leaving the session. The
code is `openreynolds/switch.py`.

## /model

```
/model                        provider, model, effort and mode; the known models here
/model <model>                another model on the same provider
/model <provider>:<model>     another provider and a model on it
/model <provider>             another provider, at its preset's default model
```

Only a real preset name counts before a colon, because model ids have colons of their
own: `/model qwen3:8b` is a model on the current provider.

The known models are a convenience for completion, not a limit. For `reynolds` they are
`claude-sonnet-5` and `claude-opus-5`; for `anthropic`, `claude-opus-5`,
`claude-sonnet-5` and `claude-haiku-4-5`; every other preset offers its own default
and desk model. `reynolds` does not offer its desk model, because the service meters
only those two for the agent. Any id the provider answers to can be typed.

### Checked before it is accepted

A candidate is probed the way `openreynolds doctor` probes a configuration: the key, the
endpoint and the model id, with an image. A typo, a missing key or a model that cannot
see images is refused at the moment you type it, with the reason, and the session
carries on on the model it had. The agent reads its own renders, so a text-only model
is refused.

### A key for another provider

A switch to another provider is allowed only when a key for it is available here:

- `reynolds` needs this machine's service key (from `openreynolds login`);
- a preset that needs no key, such as `ollama`, is always available;
- any other preset needs its key variable in the environment, for example
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

The key saved in the config file is used only for the provider it was saved with.
Without a key, `/model` refuses and names the variable to set, or
`openreynolds config --provider <name>` to set one up.

### When it takes effect

A switch is checked at once but applied between turns, so no turn is ever half one model
and half another:

- typed while the agent is idle, the new model answers your next message;
- typed mid-turn, it takes over when the current turn ends.

`/model` with nothing after it says when a pending switch will happen.

### What happens to the thread

- **Thinking blocks are dropped** from earlier assistant messages when the model
  changes. A thinking block carries a signature only the model that wrote it accepts,
  and the reasoning has already become the words and tool calls beside it.
- **The thread must fit.** If it is more than 80% of the new model's context window, it
  is refreshed on the current model first (the same refresh a long session gets), and
  the switch applies after that. The window is the one OpenReynolds knows for that
  model where it knows one (`claude-haiku-4-5` is 200,000 tokens, smaller than its
  presets' 1,000,000), and otherwise the provider's: the configured window on the same
  provider, the preset's on another. A model whose window is smaller than its
  provider's and not in that table (`MODEL_CONTEXT_WINDOWS` in `llm/presets.py`) is
  not caught; set `OPENREYNOLDS_CONTEXT_WINDOW` for it.
- **Cancelling.** `/model` naming the model you are already on cancels a pending
  switch, and any refresh it was waiting on.
- **The prompt cache is written once more** for the new model, so the first request
  after a switch costs more than the ones after it.

The study's `session.json` records the model in use. A resume does not read it back: it
starts on the model the configuration or `--model` names (in the hosted app, the one the
study is resumed with), and `/model` switches again.

### The mesh desk and the front desk

The mesh desk reads the model at each `mesh` call, so it follows the switch unless
`OPENREYNOLDS_MESHER_MODEL` pins it. The front desk, which answers quickly while the
agent works, rebuilds its client on a provider change and uses that preset's desk
model.

## /effort

```
/effort                  the current effort
/effort low|medium|high  set it
```

Effort is read on every request, so a change applies from the next one. There is
nothing to wait for and nothing in the thread to rewrite. Higher effort spends more
tokens thinking. It is the one knob that moves model cost without changing what the
agent may do.

`/effort` does not change the mesh desk's own effort (`OPENREYNOLDS_MESHER_EFFORT`).

## At the start

`--model <id>` and `--effort <level>` set both for one session, as do
`OPENREYNOLDS_MODEL` and `OPENREYNOLDS_EFFORT`. The agent's own default effort is
`high`; the hosted app starts studies at `medium`.

## In the hosted app

Under the composer, a model chooser and an effort chooser for the running session send
`/model <id>` and `/effort <level>`. The models offered are the ones this account can
use: for Reynolds' model, the two the service meters; for a key of your own, the models
that key was connected with, because a hosted session carries one provider's key. The
top bar shows the model and effort as the session reports them.

## For a program

A change arrives as a `model` event on `--output-format stream-json`, with `model`,
`effort` and `provider`. One is also sent at the start of every session.
