# YuE2

YuE2 is wired as `--family yue2 --task gen`. It generates music from lyrics and
a style prompt, with optional symbolic ABC conditioning.

## Quick Start

Default packaged GGUF layout:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "[Verse]
Soft morning light is touching the window.
I hear the city waking below.
[Chorus]
Stay with the rhythm, let it carry us home.
Sing with the sunrise, we are never alone." \
  --request-option style="English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix" \
  --request-option cot=off \
  --seed 831001 \
  --out yue2.wav \
  --log
```

The default session loads `yue2-3b-q8_0.gguf` for the main AR/NAR model and
`yue2-vae-f16.gguf` for the VAE from the model root.

## Model

| Field | Value |
|---|---|
| Family | `yue2` |
| Model directory | `models/Yue2-3B-GGUF` |
| Task | `gen` |
| Main GGUF default | `yue2-3b-q8_0.gguf` |
| VAE GGUF default | `yue2-vae-f16.gguf` |
| Required sidecars | `sidecars/yue2-model-config.json`, `sidecars/yue2-generation-config.json`, `sidecars/yue2-qwen.tiktoken`, `sidecars/yue2-vae-config.json` |
| Lyrics input | `--lyrics`; `--text` is accepted as a fallback |
| Style input | `--request-option style=<prompt>` |

## Component Selection

Select the BF16 main model:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --session-option yue2.model_gguf=yue2-3b-bf16.gguf \
  --session-option yue2.vae_gguf=yue2-vae-f16.gguf \
  --lyrics "[Verse]
Soft morning light is touching the window.
[Chorus]
Stay with the rhythm, let it carry us home." \
  --request-option style="English, pop rock, bright guitars, clean drums, warm vocal" \
  --request-option cot=off \
  --seed 831001 \
  --out yue2-bf16.wav \
  --log
```

Select the F32 VAE:

```bash
--session-option yue2.vae_gguf=yue2-vae-f32.gguf
```

The component paths are relative to `--model`; absolute paths are rejected.

## Multiple CUDA devices (experimental)

AR, NAR and VAE can use separate CUDA devices through session options
`yue2.ar_device`, `yue2.nar_device` and `yue2.vae_device`. Each defaults to `-1`,
which inherits the session's device. Explicit indices require a CUDA backend;
invalid indices fail instead of silently falling back. Components assigned to
the same device share a backend context.

This is component placement, not tensor parallelism: generation stages remain
sequential, and a single component's weights, KV cache and compute workspace must
still fit on its assigned GPU. Two 8 GB cards do not become one 16 GB device.
Q4 reduces weights but does not quantize all runtime allocations. The
`*_graph_arena_mb` options control host graph metadata arenas, not GPU memory caps.

For an RTX 2080 SUPER + Tesla P4, use the **CUDA 12** image or binary archive.
[CUDA 13 removed compilation and library support for pre-Turing architectures](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html),
including Pascal P4. Expose both GPUs to the container (`--gpus all`), then check
`audiocpp_cli --list-devices` **inside that same environment**. Device indices may
change with container visibility and `CUDA_VISIBLE_DEVICES`.

If the list identifies the 2080 SUPER as CUDA:0 and P4 as CUDA:1, append these
options to the CLI generation command above:

```bash
--device 0 \
--session-option yue2.model_gguf=yue2-3b-q4_0.gguf \
--session-option yue2.ar_device=0 \
--session-option yue2.nar_device=1 \
--session-option yue2.vae_device=0
```

For the server, start with `--backend cuda` and add the same keys to the Yue2
model entry in its configuration file:

```json
"session_options": {
  "yue2.model_gguf": "yue2-3b-q4_0.gguf",
  "yue2.ar_device": "0",
  "yue2.nar_device": "1",
  "yue2.vae_device": "0"
}
```

Cross-device AR prefix caches are copied into NAR-owned memory through bounded
host staging; this does not require NVLink or CUDA peer access. Both cache copies
exist during NAR computation, and PCIe transfer adds latency. NAR's previous
chunk graph is released before preparing another prefix. AR/NAR are released
before VAE decoding, and the preceding request's VAE is released before starting
another generation.

This configuration has CPU unit coverage for option validation and cache copies;
successful full Yue2 inference on two 8 GB GPUs has **not** been verified. Use
`--log` to capture the failing stage if one component still runs out of memory.
In a CUDA test build, an optional small two-device transfer check is available:

```bash
cmake --build <build-dir> --target yue2_device_placement_test
<build-dir>/bin/yue2_device_placement_test --cuda
```

The test requires CUDA devices 0 and 1, copies synthetic prefix data in both
directions, and does not load a model. Ordinary CTest runs use CPU only.

## ABC Conditioning

Use `cot=melody` or `cot=full` to run the symbolic route. External ABC requires
one of those modes:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "[Verse]
Write the melody over this score." \
  --request-option style="English, folk pop, acoustic guitar, steady drums" \
  --request-option cot=melody \
  --request-option abc_file=/path/to/score.abc \
  --seed 831001 \
  --out yue2-abc.wav \
  --log
```

Inline ABC can be passed with `--request-option abc=<abc text>`.

## Request Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--lyrics` | text | required | Song lyrics. |
| `--text` | text | empty | Fallback lyrics source when `--lyrics` is not supplied. |
| `--request-option style=<text>` | text | required | Music style prompt. |
| `--request-option cot=<mode>` | `off`, `melody`, `full` | `full` | Symbolic planning route. |
| `--request-option abc=<text>` | ABC text | empty | Inline ABC score; requires `cot=melody` or `cot=full`. |
| `--request-option abc_file=<path>` | path | empty | ABC score file; requires `cot=melody` or `cot=full`. |
| `--request-option semantic_codes_file=<path>` | raw int32 file | empty | Teacher-forced semantic codec IDs for parity/debug runs. |
| `--request-option nar_noise_file=<path>` | raw float32 file | empty | Teacher-forced NAR noise rows with 64 columns for parity/debug runs. |
| `--request-option cfg_scale=<f>` | `0..20` | `1.01` for `cot=off`, otherwise `1.0` | Semantic classifier-free guidance scale. |
| `--request-option num_inference_steps=<n>` | integer > 0 | `32` | NAR midpoint ODE steps. |
| `--seed <n>` | integer in `[0, 2^63)` | `831001` | Generation seed. Equivalent to `--request-option seed=<n>`. |

## Sampling Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--request-option abc_temperature=<f>` | `0..5` | `0.7` | ABC planner sampling temperature. |
| `--request-option abc_top_p=<f>` | `0..1` | `0.9` | ABC planner nucleus sampling probability. |
| `--request-option abc_top_k=<n>` | integer >= 1 | `30` | ABC planner top-k limit. |
| `--request-option abc_repetition_penalty=<f>` | float > 0 | `1.005` | ABC planner repetition penalty. |
| `--request-option abc_penalty_window=<n>` | integer >= 1 | `100` | ABC planner repetition penalty window. |
| `--request-option abc_min_tokens=<n>` | integer >= 0 | `32` | Minimum ABC planner tokens before EOS is accepted. |
| `--request-option abc_max_tokens=<n>` | integer >= `abc_min_tokens` | `4096` | Maximum ABC planner tokens. |
| `--request-option semantic_temperature=<f>` | `0..5` | `1.0` | Semantic codec sampling temperature. |
| `--request-option semantic_top_p=<f>` | `0..1` | `0.95` | Semantic codec nucleus sampling probability. |
| `--request-option semantic_top_k=<n>` | integer >= 1 | `100` | Semantic codec top-k limit. |
| `--request-option semantic_repetition_penalty=<f>` | float > 0 | `1.2` | Semantic codec repetition penalty. |
| `--request-option semantic_penalty_window=<n>` | integer >= 1 | `50` | Semantic codec repetition penalty window. |
| `--request-option semantic_min_tokens=<n>` | integer >= 0 | `200` | Minimum semantic tokens before EOS is accepted. |
| `--request-option semantic_max_tokens=<n>` | integer >= `semantic_min_tokens` | `9000` | Maximum semantic codec tokens. |

## Session Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--session-option yue2.model_gguf=<file>` | relative GGUF path | `yue2-3b-q8_0.gguf` | Main AR/NAR component. |
| `--session-option yue2.vae_gguf=<file>` | relative GGUF path | `yue2-vae-f16.gguf` | VAE component. |
| `--session-option yue2.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Shared weight storage fallback for the main model and VAE. |
| `--session-option yue2.model_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Main model weight storage override. |
| `--session-option yue2.vae_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | VAE weight storage override. |
| `--session-option yue2.model_weight_context_mb=<n>` | MiB integer >= 1 | `6144` | Main model weight context size. |
| `--session-option yue2.vae_weight_context_mb=<n>` | MiB integer >= 1 | `1536` | VAE weight context size. |
| `--session-option yue2.ar_prefill_graph_arena_mb=<n>` | MiB integer >= 1 | `4096` | AR prefill graph arena size. |
| `--session-option yue2.ar_decode_graph_arena_mb=<n>` | MiB integer >= 1 | `1536` | AR one-token decode graph arena size. |
| `--session-option yue2.nar_graph_arena_mb=<n>` | MiB integer >= 1 | `6144` | NAR acoustic flow graph arena size. |
| `--session-option yue2.vae_graph_arena_mb=<n>` | MiB integer >= 1 | `1536` | VAE decode graph arena size. |
