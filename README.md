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
