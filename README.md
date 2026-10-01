# amplifier-module-provider-litellm

Amplifier provider module that uses [litellm](https://docs.litellm.ai/) for multi-provider LLM access. Supports 100+ LLM providers through standard environment variables — zero configuration needed.

## Why?

Amplifier's built-in providers (provider-anthropic, provider-openai) each support one LLM vendor. This module uses litellm as a universal adapter, so Amplifier sessions can use **any model from any provider** without provider-specific modules.

This is especially useful when running Amplifier inside platforms like [OpenClaw](https://openclaw.ai) that already manage API keys — Amplifier automatically inherits whatever providers are configured.

## Quick Start

### Install

```bash
pip install amplifier-module-provider-litellm
# or
uv add amplifier-module-provider-litellm
```

### Configure in Amplifier

Add to your `~/.amplifier/settings.yaml`:

```yaml
config:
  providers:
    - module: provider-litellm
      source: amplifier-module-provider-litellm
      config:
        default_model: anthropic/claude-opus-4-6  # optional
        timeout: 300                                # optional
```

### Self-Hosted / Local Endpoints

For llama-server, vLLM, TGI, LocalAI, LM Studio, or any OpenAI-compatible
server, use `api_base` to point directly at it:

```yaml
config:
  providers:
    - module: provider-litellm
      source: amplifier-module-provider-litellm
      config:
        default_model: openai/my-local-model
        api_base: http://localhost:8080
        # api_key defaults to "not-needed" when api_base is set
```

Use the `openai/` model prefix. The model name after the prefix is passed
to the server as-is. No environment variables are required for local
endpoints -- `api_base` and `api_key` are fully configurable from settings.

`api_base` can also be set via the `LITELLM_API_BASE` environment variable.

### Set Environment Variables

litellm reads standard provider env vars:

| Provider | Env Var |
|----------|---------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| xAI (Grok) | `XAI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Azure OpenAI | `AZURE_API_KEY` + `AZURE_API_BASE` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| Ollama | `OLLAMA_API_BASE` (defaults to `http://localhost:11434`) |
| Together | `TOGETHER_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |

[Full list →](https://docs.litellm.ai/docs/providers)

### Model Names

Use litellm's model naming convention:

```
anthropic/claude-opus-4-6
openai/gpt-4o
gemini/gemini-2.5-pro
ollama/llama3.2
groq/llama-3.3-70b-versatile
openrouter/meta-llama/llama-3-70b
bedrock/anthropic.claude-3-sonnet-20240229-v1:0
```

### All config keys

The setup wizard prompts for `api_key`, `api_base`, and `default_model`
(reordered so credentials-first flows read naturally). Every key below is a
fully supported config key -- set it directly in `settings.yaml`.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `default_model` | string | `anthropic/claude-opus-4-6` | Canonical key. `model` is accepted as a read-alias for backwards compatibility. |
| `api_base` | string | `$LITELLM_API_BASE` | Base URL for self-hosted endpoints |
| `api_key` | string | env-var / `"not-needed"` | |
| `timeout` | float (seconds) | `300` | |
| `drop_params` | bool | `True` | Let litellm silently drop params a given provider doesn't support |
| `raw_debug` | bool | `False` | Attach a redacted copy of the litellm kwargs to the `llm:request` event |
| `overloaded_delay_multiplier` | float | `10.0` | Extra backoff multiplier applied to "overloaded" errors |
| `max_retries`, `min_retry_delay`, `max_retry_delay`, `retry_jitter` | various | `3` / `1.0` / `60.0` / `True` | **Client-side** retry policy (via amplifier-core's `retry_with_backoff`) -- separate from and not composed with litellm's own internal `num_retries` |
| `use_streaming` | bool | `True` | Set `False` to force non-streaming completions |
| `extra_request_params` | dict | `{}` | Merged last into the kwargs passed to `litellm.acompletion()` -- an escape hatch for any litellm-native kwarg not listed above (e.g. `top_p`, `seed`, `frequency_penalty`) |
| `priority` | int | n/a | Read by the orchestrator's provider-selection logic, not by this module directly |

Boolean and numeric keys accept native types or the string forms a config
wizard writes (`"true"`/`"false"`, `"300"`); invalid numeric strings warn
and fall back to the default rather than crashing at mount. Unrecognized
config keys produce a mount-time warning (with a did-you-mean suggestion)
rather than a silent no-op. The `debug` key (documented in older versions)
is never wired to anything -- use `raw_debug`.

## How It Works

1. Amplifier session requests an LLM completion
2. `provider-litellm` receives the `ChatRequest`
3. Converts to litellm format and calls `litellm.acompletion()`
4. litellm routes to the correct provider based on model name prefix
5. Response is converted back to Amplifier's `ChatResponse`

No bridge services, no proxy servers, no credential duplication. Just env vars and a model name.

## Use with OpenClaw

When Amplifier runs as a sidecar inside OpenClaw, all of OpenClaw's configured API keys are available as environment variables. This means:

- **Zero config**: provider-litellm inherits OpenClaw's credentials automatically
- **Any model**: whatever providers the user has configured in OpenClaw "just work"
- **Community friendly**: users with only Ollama (free, local) get full Amplifier access

```
User configures OpenClaw with Ollama
  → OpenClaw sets OLLAMA_API_BASE env var
    → Amplifier's provider-litellm picks it up
      → Amplifier sessions use Ollama
        → No API keys needed, no cost
```

## Development

```bash
git clone https://github.com/microsoft/amplifier-module-provider-litellm
cd amplifier-module-provider-litellm
uv sync --dev
uv run pytest -v
```


## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
## License

MIT — see [LICENSE](LICENSE).

### Completion waits

Healthy model requests have no default read/elapsed deadline. Explicit `timeout`
configuration remains supported; connection and pool acquisition stay bounded.
Cancellation propagates. Slow generation is not evidence of a broken service.
Per-call options can be supplied in `request_options`; explicit keyword arguments
take precedence over that mapping.

`raw: true` records the full redacted SDK request on `llm:request.raw`; the legacy `raw_debug` option remains an alias. Capture is off by default.
