#!/usr/bin/env python3
"""Package the already-built /app from a CUDA runtime image; never compile or use a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

BINARIES = ('audiocpp_cli', 'audiocpp_server', 'audiocpp_model_manager', 'model_perf')
# These belong to the host OS or installed NVIDIA driver, not the CUDA toolkit.
HOST_LIBRARIES = {
    'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1',
    'libresolv.so.2', 'libutil.so.1', 'libanl.so.1', 'ld-linux-x86-64.so.2',
    'libcuda.so.1', 'libnvidia-ml.so.1',
}


def dependencies(path, library_path=None):
    env = {**os.environ, 'LC_ALL': 'C'}
    if library_path is not None:
        env['LD_LIBRARY_PATH'] = str(library_path)
    result = subprocess.run(['ldd', str(path)], capture_output=True, text=True,
                            env=env)
    if result.returncode:
        raise RuntimeError(f'Cannot inspect dependencies of {path}: {result.stderr or result.stdout}')
    for line in result.stdout.splitlines():
        name, separator, target = line.strip().partition(' => ')
        if not separator:
            continue  # vDSO and the host dynamic loader have no => entry.
        target = re.sub(r'\s+\(0x[0-9a-fA-F]+\)$', '', target)
        if name in HOST_LIBRARIES:
            continue
        if target == 'not found':
            raise RuntimeError(f'Missing runtime dependency {name} required by {path}')
        if not Path(target).is_absolute() or not Path(target).is_file():
            raise RuntimeError(f'Invalid dependency path for {name}: {target}')
        yield name, Path(target)


def package(args):
    if not re.fullmatch(r'v[A-Za-z0-9_.-]{1,115}', args.version):
        raise ValueError('version must be a Docker-compatible v-prefixed tag')
    source = args.app_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f'audio-{args.version}-bin-linux-amd64-{args.backend}'
    with tempfile.TemporaryDirectory(prefix='audiocpp-package-', dir=output) as work:
        bundle = Path(work) / name
        (bundle / 'bin').mkdir(parents=True)
        (bundle / 'lib').mkdir()
        roots = []
        for binary in BINARIES:
            path = source / binary
            if not path.is_file():
                raise FileNotFoundError(f'Image is missing {binary}')
            shutil.copy2(path, bundle / 'bin' / binary)
            roots.append(path)
            launcher = bundle / binary
            launcher.write_text(
                '#!/bin/sh\nset -eu\n'
                'bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
                'export LD_LIBRARY_PATH="$bundle_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
                f'exec "$bundle_dir/bin/{binary}" "$@"\n')
            launcher.chmod(0o755)
        for directory in ('model_specs', 'tools'):
            shutil.copytree(source / directory, bundle / directory)
        shutil.copy2(args.source_dir / 'LICENSE', bundle / 'LICENSE')
        # Collect all project backends, including libraries loaded via dlopen.
        libraries = {}
        for path in source.glob('*.so*'):
            if path.is_file():
                libraries[path.name] = path
                roots.append(path)
        # ldd walks transitive DT_NEEDED entries. Inspect the original paths while
        # the runtime image's loader configuration and library locations are intact.
        for root in roots:
            for soname, path in dependencies(root):
                existing = libraries.get(soname)
                if existing and existing.resolve() != path.resolve():
                    raise RuntimeError(f'Conflicting runtime libraries for {soname}')
                libraries[soname] = path
        for soname, path in sorted(libraries.items()):
            if soname not in HOST_LIBRARIES:
                shutil.copy2(path, bundle / 'lib' / soname)  # Dereference image symlinks.
        # Check the relocated executables and plugins using only the bundled
        # search path. A dependency accidentally left in /app or the image's
        # system directories must not make the archive look self-contained.
        for path in [*(bundle / 'bin').iterdir(), *(bundle / 'lib').iterdir()]:
            for soname, resolved in dependencies(path, bundle / 'lib'):
                if not resolved.resolve().is_relative_to(bundle):
                    raise RuntimeError(f'Unbundled dependency after relocation: {soname} => {resolved}')
        # Preserve installed runtime notices when provided by the image.
        notices = bundle / 'licenses'
        notices.mkdir()
        for path in Path('/usr/share/doc').glob('*/copyright'):
            # CUDA redistribution notices and the system runtime libraries copied above.
            if any(word in path.parent.name for word in ('cuda', 'cublas', 'gcc', 'libstdc++', 'libgomp')):
                shutil.copy2(path, notices / (path.parent.name + '-copyright'))
        for path in Path('/usr/local').glob('cuda*/**/EULA*'):
            if path.is_file():
                shutil.copy2(path, notices / ('cuda-' + path.name))
        metadata = {
            'version': args.version, 'revision': args.revision,
            'backend': args.backend, 'cuda_version': args.cuda_version,
            'platform': 'linux/amd64', 'base_system': 'Ubuntu 24.04 (glibc 2.39)',
            'libraries': sorted(libraries),
        }
        (bundle / 'build-info.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (bundle / 'README.txt').write_text(f'''audio.cpp {args.version} — Linux amd64 {args.backend} (CUDA {args.cuda_version})

This archive reuses the exact binaries built for the corresponding Docker image.
Run the launchers in this directory, which locate the bundled shared libraries:

  ./audiocpp_server --config /absolute/path/to/server.json
  ./audiocpp_cli --help
  ./audiocpp_model_manager --help
  ./model_perf --help

Requirements:
- Linux x86_64, Ubuntu 24.04 / glibc 2.39 or a compatible newer system.
- An NVIDIA GPU and a driver compatible with CUDA {args.cuda_version}.
  The host NVIDIA driver and glibc are not bundled. Docker and nvcc are not required.
- Install ffmpeg for MP3/FLAC/M4A/OGG/WebM uploads, plus python3, curl and
  ca-certificates for the optional model-management tools (Ubuntu: apt-get install
  ffmpeg python3 curl ca-certificates).
- Download model weights separately and provide a server configuration.

bin/ contains the original executables. Use the top-level launchers so lib/ is
found. model_specs/ and tools/ are copied from the image. build-info.json records
versions and the runtime libraries included. Third-party libraries retain their
respective licenses; see licenses/. This is not a statically linked binary bundle.
''')
        archive = output / (name + '.tar.gz')
        # Keep partial archives outside the final artifact name if packaging fails.
        partial = Path(work) / archive.name
        with tarfile.open(partial, 'w:gz', compresslevel=6) as tar:
            tar.add(bundle, arcname=name)
        if partial.stat().st_size >= 2 * 1024**3:
            raise RuntimeError('Binary archive exceeds the GitHub Release per-file size limit')
        partial.replace(archive)
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        archive.with_suffix(archive.suffix + '.sha256').write_text(f'{digest.hexdigest()}  {archive.name}\n')
        print(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-dir', type=Path, default=Path('/app'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--backend', choices=('cuda12', 'cuda13'), required=True)
    parser.add_argument('--cuda-version', required=True)
    parser.add_argument('--revision', required=True)
    package(parser.parse_args())


if __name__ == '__main__':
    main()
