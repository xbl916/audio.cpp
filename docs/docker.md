# Running audio.cpp in Docker

## Table of Contents

- [Prerequisites](#prerequisites)
- [Image Variants](#image-variants)
- [Published Images](#published-images)
- [Build Images locally](#build-images-locally)
- [Usage](#usage)
- [Examples](#examples)

## Prerequisites

- Docker must be installed and running on your system.
- For CUDA:
  - The [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) must be installed.
- For Vulkan:
  - The host must expose a working Vulkan device to Docker, typically through `/dev/dri` on Linux.
  - The container user needs access to the host render/video device groups.

## Image Variants

The following image variants are available:

- **full**: Provides the main tools **cli** and **server** and test binaries in one image. When running the container, the first argument selects the tool to execute.

The following backends are supported:
- **cuda12**
- **cuda13**
- **vulkan**
- **cpu**

The following architectures are supported:
- **amd64**
- **arm64**

## Published Images

Docker images are published when a `v*` Git tag is pushed or through the
**Build and publish Docker images** workflow's manual trigger. Normal branch pushes,
pull requests, and timers do not start builds in this fork. Tag pushes automatically
build only these two targets:

| Backend | Architecture | Image |
| --- | --- | --- |
| CUDA 12 | linux/amd64 | `ghcr.io/xbl916/audio.cpp:full-cuda12` |
| CUDA 13 | linux/amd64 | `ghcr.io/xbl916/audio.cpp:full-cuda13` |

The workflow publishes to `ghcr.io/<owner>/<repository>` in lowercase. CPU, Vulkan,
and arm64 builds are available on demand through the manual inputs below.

Images for a specific day/commit can be found in the
[versions](https://github.com/xbl916/audio.cpp/pkgs/container/audio.cpp/versions?filters%5Bversion_type%5D=tagged)
history. The format is `full-<backend>-<date>-<shortsha>`, for example
`full-cuda12-20260725-db7d2c4`.

Pushing a version tag such as `v1.2.3` also publishes `full-cuda12-v1.2.3` and
`full-cuda13-v1.2.3`, while updating the usual `full-cuda12` and `full-cuda13` tags.
The binaries embed version `1.2.3`. Every tag push builds the selected commit.
Use Docker-compatible version tag names containing only letters, digits, underscores,
dots, and hyphens (at most 116 characters).

After committing the changes you want to release, create and push a new tag:

```bash
git tag v1.2.3
git push origin v1.2.3
# After the Docker workflow succeeds:
docker pull ghcr.io/xbl916/audio.cpp:full-cuda12-v1.2.3
```

### Manual builds for other targets

Use `gh workflow run` after authenticating GitHub CLI. The `backends` input accepts
`cpu,cuda12,cuda13,vulkan`; `architectures` accepts `amd64,arm64`. Values are
comma-separated and every selected backend is built for every selected architecture.
Defaults are `cuda12,cuda13` and `amd64`. Manual runs always build the selected ref.

```bash
# CPU and Vulkan, amd64 only, from the current main branch:
gh workflow run docker.yml --repo xbl916/audio.cpp --ref main \
  -f backends=cpu,vulkan -f architectures=amd64

# Both CUDA versions, with amd64 and arm64 images:
gh workflow run docker.yml --repo xbl916/audio.cpp --ref main \
  -f backends=cuda12,cuda13 -f architectures=amd64,arm64

# CPU, arm64 only:
gh workflow run docker.yml --repo xbl916/audio.cpp --ref main \
  -f backends=cpu -f architectures=arm64

# All four backends on both architectures (the previous full build):
gh workflow run docker.yml --repo xbl916/audio.cpp --ref main \
  -f backends=cpu,cuda12,cuda13,vulkan -f architectures=amd64,arm64

# Build selected targets from a version tag that contains this workflow:
gh workflow run docker.yml --repo xbl916/audio.cpp --ref v1.2.3 \
  -f backends=cpu,vulkan -f architectures=amd64,arm64
```

A tag ref supplies the embedded version and creates versioned image tags for the
selected backends. A branch ref defaults to embedded version `dev`; use
`-f version=1.2.3` to change that embedded version only. Each published backend's
manifest contains exactly the selected architectures, replacing the previous manifest
for the same image tag. Unselected backends are not updated.

The selected ref must contain the updated workflow. For example, the existing
`v0.7.3` tag still contains the previous workflow, which builds all eight combinations;
use `main` or a newer release tag for these selection inputs.

### Checking the trigger

The tag must be pushed to GitHub; creating it locally does not start a build. Check
the workflow's event in Actions: `push` means an automatic tag build, while
`workflow_dispatch` means a manual run. A successful manual run verifies the build
but does not verify that the push trigger fired.

### Optional compile checks

The Windows, Linux, macOS, Nix, and server memory checks are manual-only. They do
not publish Docker images. Run a specific check from the current main branch:

```bash
gh workflow run windows-build.yml --repo xbl916/audio.cpp --ref main
gh workflow run linux-build.yml --repo xbl916/audio.cpp --ref main
gh workflow run mac-build.yml --repo xbl916/audio.cpp --ref main
gh workflow run nix-build.yml --repo xbl916/audio.cpp --ref main
gh workflow run server-memory-guard.yml --repo xbl916/audio.cpp --ref main
```

### Optional binary release packages

Tag pushes only trigger Docker builds. The separate **Binary release (manual)**
workflow builds native Windows/Linux/macOS packages and must be started explicitly.
It does not build Docker images. Use a ref containing this updated workflow:

```bash
# Build native packages without publishing a GitHub Release:
gh workflow run release.yml --repo xbl916/audio.cpp --ref main \
  -f version=1.2.3 -f publish=false

# Build and publish native packages for a release tag:
gh workflow run release.yml --repo xbl916/audio.cpp --ref v1.2.3 \
  -f version=1.2.3 -f publish=true
```

The `publish` input defaults to false, including when a tag ref is selected.
Existing tags keep the workflow files they originally contained; `v0.7.3-2` still
contains the old automatic binary-release trigger. New tags must include the
commit that makes binary releases manual-only.


## Build Images locally

If you would like to build the images locally, you can use the available
Dockerfiles in `.devops`.

### CUDA

Build with the default CUDA 12.x version. See `.devops/cuda.Dockerfile`.

```bash
docker build -f .devops/cuda.Dockerfile -t local/audio.cpp:full-cuda12 .
```

Build with a specific CUDA version, for example 13.3.0:

```bash
docker build -f .devops/cuda.Dockerfile -t local/audio.cpp:full-cuda13 --build-arg CUDA_VERSION=13.3.0 .
```

Build for a specific set of GPU architectures (e.g. for faster, less portable builds):

```bash
docker build -f .devops/cuda.Dockerfile -t local/audio.cpp:full-cuda12 --build-arg CUDA_DOCKER_ARCH="86;89" .
```

### Vulkan

```bash
docker build -f .devops/vulkan.Dockerfile -t local/audio.cpp:full-vulkan .
```

### CPU

```bash
docker build -f .devops/cpu.Dockerfile -t local/audio.cpp:full-cpu .
```

## Usage

For CLI use, mount the model directory `<models-dir>` into the container.
An additional `<output-dir>` should be mounted for tasks that write files.

### CUDA

```bash
docker run --rm --gpus all -v "<models-dir>:/models:ro" ghcr.io/xbl916/audio.cpp:full-cuda12 <cli|server> --model /models/<model> <...>
```

### Vulkan

```bash
docker run --rm --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  -v "<models-dir>:/models:ro" \
  ghcr.io/xbl916/audio.cpp:full-vulkan \
  <cli|server> --backend vulkan --model /models/<model> <...>
```

### Native WebUI

For the native WebUI with model downloads and dynamic model management, mount a
writable model directory and expose the server port:

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v "<models-dir>:/app/models" \
  ghcr.io/xbl916/audio.cpp:full-cuda12 \
  server --ui --ui-management --host 0.0.0.0 --port 8080 --backend cuda
```

For Vulkan, expose the host render device and use the Vulkan backend:

```bash
docker run --rm --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  -p 8080:8080 \
  -v "<models-dir>:/app/models" \
  ghcr.io/xbl916/audio.cpp:full-vulkan \
  server --ui --ui-management --host 0.0.0.0 --port 8080 --backend vulkan
```

Open `http://127.0.0.1:8080` on the host. Use a writable mount when the UI
should download or prepare models. For a read-only model directory, omit
`--ui-management` or mount the directory as read-only and load only models that
already exist in the configured path.

### CPU

```bash
docker run --rm -v "<models-dir>:/models:ro" ghcr.io/xbl916/audio.cpp:full-cpu <cli|server> --model /models/<model> <...>
```

See the fully working [examples](#examples) below.

## Examples

Examples for Docker, including CUDA and CPU, are available in `examples/docker`.

### CLI

The **[examples](../examples/docker/cli/EXAMPLE.md)** in `examples/docker/cli`
demonstrate how to run the audio.cpp CLI with `docker run`. The examples include:

- **PocketTTS:** Text-to-Speech
- **Qwen3-TTS:** Text-to-Speech with Voice Cloning

### Server

The **[examples](../examples/docker/server/EXAMPLE.md)** in `examples/docker/server`
demonstrate how to run the audio.cpp server with `docker compose`. The examples include:

- **PocketTTS:** Text-to-Speech
- **Qwen3-TTS:** Text-to-Speech with Voice Cloning
