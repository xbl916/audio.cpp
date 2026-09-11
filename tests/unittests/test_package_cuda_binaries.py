"""Exercise binary packaging with real ELF dependencies, without Docker or a GPU."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'package_cuda_binaries', Path(__file__).resolve().parents[2] / 'tools/package_cuda_binaries.py')
PACKAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGER)


@unittest.skipUnless(shutil.which('cc') and shutil.which('ldd'), 'requires a Linux C toolchain')
class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="package test ' space-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / 'app'
        self.external = self.root / 'external'
        self.source = self.root / 'source'
        for folder in (self.app, self.external, self.source, self.app / 'model_specs', self.app / 'tools'):
            folder.mkdir()
        (self.source / 'LICENSE').write_text('fixture license\n')
        (self.app / 'model_specs' / 'fixture.json').write_text('{}')
        (self.app / 'tools' / 'fixture.py').write_text('# fixture')
        (self.root / 'external.c').write_text('int external_value(void) { return 7; }')
        (self.root / 'project.c').write_text('extern int external_value(void); int value(void) { return external_value(); }')
        (self.root / 'main.c').write_text('#include <stdio.h>\nextern int value(void); int main(void) { printf("value=%d\\n", value()); return value() != 7; }')
        self.cc('-shared', '-fPIC', self.root / 'external.c', '-Wl,-soname,libfixtureexternal.so', '-o', self.external / 'libfixtureexternal.so')
        self.cc('-shared', '-fPIC', self.root / 'project.c', '-L' + str(self.external), '-lfixtureexternal', '-o', self.app / 'libfixtureproject.so')
        self.cc(self.root / 'main.c', '-L' + str(self.app), '-lfixtureproject', '-Wl,-rpath-link,' + str(self.external), '-o', self.app / 'audiocpp_server')
        for name in PACKAGER.BINARIES:
            if name != 'audiocpp_server':
                shutil.copy2(self.app / 'audiocpp_server', self.app / name)
        self.args = argparse.Namespace(app_dir=self.app, output_dir=self.root / 'dist', source_dir=self.source,
                                       version='v1.2.3', backend='cuda12', cuda_version='12.9.2', revision='fixture')
        self.env = {'LD_LIBRARY_PATH': str(self.app) + ':' + str(self.external)}

    def cc(self, *args):
        subprocess.run(['cc', *map(str, args)], check=True, capture_output=True)

    def test_archive_runs_after_relocation(self):
        with patch.dict(os.environ, self.env):
            PACKAGER.package(self.args)
        archive = next(self.args.output_dir.glob('*.tar.gz'))
        checksum = archive.with_suffix('.gz.sha256').read_text().split()[0]
        self.assertEqual(checksum, hashlib.sha256(archive.read_bytes()).hexdigest())
        extracted = self.root / 'new location'
        extracted.mkdir()
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(extracted)], check=True)
        bundle = next(extracted.iterdir())
        shutil.rmtree(self.app)
        shutil.rmtree(self.external)
        for name in PACKAGER.BINARIES:
            result = subprocess.run([str(bundle / name), '--help'], capture_output=True, text=True, cwd=extracted)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, 'value=7\n')
        self.assertTrue((bundle / 'lib' / 'libfixtureproject.so').is_file())
        self.assertTrue((bundle / 'lib' / 'libfixtureexternal.so').is_file())
        self.assertFalse((bundle / 'lib' / 'libc.so.6').exists())
        self.assertTrue((bundle / 'model_specs' / 'fixture.json').is_file())
        self.assertTrue((bundle / 'README.txt').is_file())
        self.assertEqual(len(list(self.args.output_dir.iterdir())), 2)

    def test_missing_dependency_fails_without_archive(self):
        (self.external / 'libfixtureexternal.so').unlink()
        with patch.dict(os.environ, self.env), self.assertRaisesRegex(RuntimeError, 'Missing runtime dependency'):
            PACKAGER.package(self.args)
        self.assertEqual(list(self.args.output_dir.iterdir()), [])

    def test_missing_binary_fails_without_archive(self):
        (self.app / 'audiocpp_cli').unlink()
        with self.assertRaises(FileNotFoundError):
            PACKAGER.package(self.args)
        self.assertEqual(list(self.args.output_dir.iterdir()), [])

    def test_invalid_version_is_rejected(self):
        self.args.version = '../escape'
        with self.assertRaises(ValueError):
            PACKAGER.package(self.args)
        self.assertFalse(self.args.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
