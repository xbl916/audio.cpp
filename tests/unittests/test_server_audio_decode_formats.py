"""Exercise the server's decoder with real FFmpeg containers, without a model/GPU."""
import concurrent.futures
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import wave


def run(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, **kwargs)


def inspect_wav(path):
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    pos = 12
    fmt = payload = None
    while pos + 8 <= len(data):
        kind, size = struct.unpack_from("<4sI", data, pos)
        chunk = data[pos + 8:pos + 8 + size]
        if kind == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", chunk)
        elif kind == b"data":
            payload = chunk
        pos += 8 + size + (size % 2)
    assert fmt and payload
    _, channels, rate, _, _, bits = fmt
    assert channels == 2 and rate == 44100, fmt
    duration = len(payload) / (rate * channels * bits / 8)
    assert 0.9 < duration < 1.2, duration
    assert any(payload), "silent output"


def main():
    decoder, ffmpeg = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="audio-decode-formats-") as root:
        root = Path(root)
        # Shell metacharacters in TMPDIR exercise argv-based process launching.
        temp = root / "temp ' $() ` ; space"
        temp.mkdir()
        env = dict(os.environ, TMPDIR=str(temp))
        source = root / "source.wav"
        with wave.open(str(source), "wb") as wav:
            wav.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            wav.writeframes(b"".join(struct.pack("<hh", int(10000 * math.sin(i * 0.06)),
                                                  int(8000 * math.sin(i * 0.09)))
                                     for i in range(44100)))
        cases = [("mp3", "libmp3lame"), ("flac", "flac"), ("m4a", "aac"),
                 ("aac", "aac"), ("ogg", "libvorbis"), ("opus", "libopus"),
                 ("webm", "libopus"), ("aiff", "pcm_s16be"), ("caf", "pcm_s16le"),
                 ("wav", "adpcm_ima_wav")]
        uploads = []
        for suffix, codec in cases:
            encoded = root / ("encoded." + suffix)
            run([ffmpeg, "-v", "error", "-y", "-i", str(source), "-c:a", codec, str(encoded)])
            # Strip any useful extension, just as clients with wrong names/MIMEs do.
            upload = root / (suffix + ".upload")
            upload.write_bytes(encoded.read_bytes())
            uploads.append((suffix, upload))

        def check(case):
            suffix, upload = case
            output = root / (suffix + ".decoded.wav")
            run([decoder, str(upload), str(output)], env=env)
            if suffix in ("opus", "webm"):
                # Opus itself uses 48 kHz, so compare against the encoded rate.
                run([ffmpeg, "-v", "error", "-i", str(output), "-ar", "44100",
                             "-c:a", "pcm_f32le", str(root / (suffix + ".44100.wav"))])
                inspect_wav(root / (suffix + ".44100.wav"))
                data = output.read_bytes()
                pos = data.index(b"fmt ")
                assert struct.unpack_from("<I", data, pos + 12)[0] == 48000
            else:
                inspect_wav(output)
            return suffix

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            print("decoded:", ", ".join(pool.map(check, uploads)))
        native = root / "native.out.wav"
        run([decoder, str(source), str(native)], env=env)
        assert source.read_bytes() == native.read_bytes()
        for index, data in enumerate((b"garbage", b"RIFF\xff\xff\xff\xffWAVE", b"#EXTM3U\nhttp://127.0.0.1/private\n",
                                     b"ffconcat version 1.0\nfile '/etc/passwd'\n")):
            invalid = root / f"invalid{index}.wav"
            invalid.write_bytes(data)
            result = subprocess.run([decoder, str(invalid), str(native)], env=env, capture_output=True)
            assert result.returncode != 0 and b"400:" in result.stderr, result.stderr
        # Verify failures from the external process without relying on huge fixtures.
        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake = fake_bin / "ffmpeg"
        fake_env = dict(env, PATH=str(fake_bin))
        failures = [
            ("raise SystemExit(7)", b"400:"),
            ("open(sys.argv[-1], 'wb').close()", b"400:"),
            ("open(sys.argv[-1], 'wb').write(b'x' * 100)", b"400:"),
            ("f = open(sys.argv[-1], 'wb'); f.truncate(256 * 1024 * 1024); f.close()", b"413:"),
            ("time.sleep(120)", b"400: audio decoding exceeded"),
        ]
        for script, error in failures:
            fake.write_text(f"#!{sys.executable}\nimport sys, time\n{script}\n")
            fake.chmod(0o700)
            start = time.monotonic()
            result = subprocess.run([decoder, str(uploads[0][1]), str(native)],
                                    env=fake_env, capture_output=True, timeout=70)
            assert result.returncode != 0 and error in result.stderr, result.stderr
            if "sleep" in script:
                assert 59 <= time.monotonic() - start < 70
            assert list(temp.iterdir()) == [], "temporary decoding files leaked on failure"
        assert list(temp.iterdir()) == [], "temporary decoding files leaked"
        print("native WAV, invalid inputs, concurrency, size limits, timeout, and cleanup passed")


if __name__ == "__main__":
    main()
