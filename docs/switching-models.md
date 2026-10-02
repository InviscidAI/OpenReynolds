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
`gpt-6.1-sol` (the default), `gpt-6-astra`, `claude-sonnet-5-5` and `claude-opus-5-5`;
for `anthropic`, `claude-opus-5`, `claude-sonnet-5` and `claude-haiku-4-5`; every other
preset offers its own default and desk model. `reynolds` does not offer its desk models,
because the service meters only those four for the agent. Any id the provider answers
to can be typed.

### Two families behind `reynolds`

The workspace service serves GPT and Claude, and the model picks the API: a `claude-` id
goes over Anthropic's Messages API at `{service}/v1/llm`, anything else over OpenAI's
Responses API at `{service}/v1/llm/v1`, both on the service key (which cloud the service
relays each to is its business). So `/model claude-opus-5-5` on a GPT session is a
switch of client as well as of model, even though the provider, key and service are the
same: the client is rebuilt at the next turn, the front desk moves from `gpt-6-luna` to
`claude-haiku-4-5` (it always speaks the main model's family), and the window moves from
GPT's 272,000 -- where those models start billing at a long-context rate the service
does not price -- to Claude's 1,000,000. A switch within a family keeps the client.

The thread carries over as it does between providers: each turn is replayed to the new
API as its words and tool calls, and the earlier model's reasoning -- GPT's encrypted
reasoning items, Claude's signed thinking blocks -- is dropped, because only the model
that wrote it can read it. The CAD desk and its reviewer follow at their next call.

A request a model's safety classifier declines is retried on the other model of the
same family -- `gpt-6.1-sol` and `gpt-6-astra`, `claude-sonnet-5-5` and
`claude-opus-5-5` -- and never across families.

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

### Resuming

The study's `session.json` records three things: the model in use, the provider that
served it, and the endpoint it was served from (`llm_base_url` as configured, empty for
the preset's own). `--study <id>` carries on on that record -- the model `/model` last
left the study on, even when the configured default has changed since -- unless
`--model` or `OPENREYNOLDS_MODEL` names one, which wins here as it does at the start.

The record is restored whole or not at all, and the terminal's test is whether this
machine can serve it:

- **The provider must be reachable here**, by the same rule `/model` applies to a
  switch: a key for it in the environment, this machine's service key for `reynolds`,
  or a preset that needs no key. A record naming another provider than the configured
  one is restored by switching to it, bringing that provider's key, endpoint, context
  window and desk model with it, exactly as `/model <provider>:<model>` does.
- **On the configured provider, the endpoint must match too.** A provider name is not
  an endpoint: two keys of one family, a vendor's own and a gateway or router in front
  of it, answer to different model ids, so `anthropic/claude-sonnet-4.5` restored onto a
  direct Anthropic key would be accepted here and refused by the vendor mid-turn with a
  400 about a model that does not exist. The stored endpoint is compared with this run's
  configured one, and a study recorded before the endpoint was kept has none: that is
  refused rather than guessed at.

When the record cannot be honoured, this run falls back to the configured model and says
so in one line. **The study keeps what it recorded.** The run that could not serve the
pair is the last one that should forget it, so nothing is overwritten and a resume where
that key is set carries on there.

Two limits worth knowing:

- **A study resumed where it has never run starts on the configured model.** The record
  is the local `session.json`, and a study opened in the browser or last run on another
  laptop has none here; what the platform is asked for on a resume (`_recover_session`)
  is the study's home and its id, not its model.
- **`OPENREYNOLDS_PROVIDER` is not an explicit signal.** Only `--model` and
  `OPENREYNOLDS_MODEL` count as somebody asking for this model, so naming a provider on
  its own does not hold a resume to it: the recorded pair is restored over it. Name a
  model as well to move a study.

A model id does not name a provider on its own -- `claude-opus-5` was valid on
`anthropic` and on `reynolds` both, and OpenRouter's ids look like Anthropic's -- so a
study recorded before the provider was kept names a model and no provider, and starts on
the configured model as it always did.

`reynolds` is the one provider whose list is closed, because it served Claude 5 until
2026-10 and serves GPT and the Claude 5.5 models now. A study recorded on `reynolds`
with a model the service no longer meters is not restored as recorded, which would be a
4xx on the first turn: it carries on on `reynolds` -- `claude-sonnet-5` on
`claude-sonnet-5-5` and `claude-opus-5` on `claude-opus-5-5`, because whoever chose
Claude chose Claude, and anything else on the default -- says so, and the record is
rewritten. A config file saved then loads the same way.

### The mesh desk and the front desk

The mesh desk reads the model at each `mesh` call, so it follows the switch unless
`OPENREYNOLDS_MESHER_MODEL` pins it. The front desk, which answers quickly while the
agent works, rebuilds its client on a provider change -- or on `reynolds`, a change of
family -- and uses that provider's (or family's) desk model.

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

`--model <id>` and `--effort <level>` set both for the session, as do
`OPENREYNOLDS_MODEL` and `OPENREYNOLDS_EFFORT`, and either beats what the study has
stored. The model is recorded on the study with the provider and the endpoint that
served it, so the one named here is also the one a later resume carries on with.
`OPENREYNOLDS_PROVIDER` is not one of these: it sets which provider a new session
starts on, and a resume still restores the provider the study recorded. The agent's own
default effort is `high`; the hosted app starts studies at `medium`.

## In the hosted app

Under the composer, a model chooser and an effort chooser for the running session send
`/model <id>` and `/effort <level>`. The models offered are the ones this account can
use: for Reynolds' model, the four the service meters; for a key of your own, the models
that key was connected with, because a hosted session carries one provider's key. The
top bar shows the model and effort as the session reports them.

Resuming a study there reads the same record, and applies its own half of the gate: the
key the resume is starting with must belong to the provider that served the model and to
the same endpoint as the record, and must be able to answer to that id (Reynolds' model
is metered rather than a key, so it serves only the four models the service prices; a key
of your own serves any id its vendor answers to). A record with no endpoint is refused
rather than guessed at. That is a narrower test than the terminal's: the app never
switches provider to honour a record, because a hosted session carries one key. When it
fails, the study starts on the account's configured model, with nothing on the page to
say so.

## For a program

A change arrives as a `model` event on `--output-format stream-json`, with `model`,
`effort` and `provider`. One is also sent at the start of every session.
