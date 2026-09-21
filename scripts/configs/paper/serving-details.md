# Endpoint context and client input settings

These values describe the model endpoints and client metadata used in the paper
evaluations. For a self-hosted or managed endpoint, configure or select an
equivalent context capacity. The model YAMLs do not configure the serving process.

| Model | Endpoint context window used (tokens) | Client input capacity declared (tokens) |
| --- | ---: | ---: |
| Claude Opus 5 / Sonnet 5 | Provider-managed | 1,000,000 |
| Gemini 3.6 Flash | Provider-managed | 1,048,576 |
| GPT-5.6 Sol / Luna / Terra | Provider-managed | No client override |
| Kimi K3 | 1,048,576 | 984,576 |
| GLM-5.2 FP8 | 400,000 | 268,928 |
| DeepSeek V4 Flash 0731 | 1,048,576 | 1,000,000 |
| Laguna S 2.1 | 1,048,576 | 1,048,576 |
| Inkling Small | 262,144 | 198,144 |

**Endpoint context window** is the capacity available for the input and generated
response together. Its configuration depends on the hosting service. The maximum
output budget may not fit alongside a maximum-length input.

**Client input capacity declared** is model metadata supplied to LiteLLM in the
paper runs. It does not enlarge the endpoint's context window or necessarily
enforce input truncation. The YAMLs in this folder configure requests; they do not
install those registry overrides. An endpoint using a custom deployment name may
need additional metadata to match the paper client setup.
