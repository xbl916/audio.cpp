#include "audio_decode.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace minitts::server {
namespace {
constexpr std::size_t max_audio_bytes = 256 * 1024 * 1024;

// Only bypass FFmpeg for encodings the native WAV reader understands. Compressed
// WAV (e.g. ADPCM) and other containers still go through content-based probing.
bool native_wav(std::string_view bytes) {
    if (bytes.size() < 12 || bytes.substr(0, 4) != "RIFF" || bytes.substr(8, 4) != "WAVE") {
        return false;
    }
    auto u16 = [&](std::size_t i) {
        return static_cast<unsigned char>(bytes[i]) |
            (static_cast<unsigned char>(bytes[i + 1]) << 8);
    };
    bool supported_format = false;
    bool has_audio = false;
    for (std::size_t pos = 12; bytes.size() - pos >= 8;) {
        const std::uint32_t size = static_cast<std::uint32_t>(u16(pos + 4)) |
            (static_cast<std::uint32_t>(u16(pos + 6)) << 16);
        const auto data = pos + 8;
        if (size > bytes.size() - data) return false;
        if (bytes.substr(pos, 4) == "fmt " && size >= 16) {
            int format = u16(data);
            const int bits = u16(data + 14);
            if (format == 0xfffe) {
                const std::string_view guid_tail("\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71", 14);
                if (size < 40 || u16(data + 16) < 22 || bytes.substr(data + 26, 14) != guid_tail) return false;
                format = u16(data + 24);
            }
            if (u16(data + 2) == 0 || (u16(data + 4) == 0 && u16(data + 6) == 0)) return false;
            supported_format = (format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
                (format == 3 && (bits == 32 || bits == 64)) ||
                ((format == 6 || format == 7) && bits == 8);
        }
        if (bytes.substr(pos, 4) == "data" && size > 0) has_audio = true;
        const auto next = data + size;
        if (next == bytes.size()) break;
        pos = next + (size & 1u);
    }
    return supported_format && has_audio;
}

#ifndef _WIN32
struct TemporaryAudio {
    std::filesystem::path directory;
    TemporaryAudio() {
        auto pattern = (std::filesystem::temp_directory_path() / "audiocpp-decode-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        if (!mkdtemp(name.data())) throw AudioDecodeError(500, "could not create audio decoding directory");
        directory = name.data();
    }
    ~TemporaryAudio() {
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }
};

void run_ffmpeg(const std::string & input, const std::string & output) {
    // No shell, client filenames, playlists, or network protocols. A seekable
    // private input file also handles M4A files with their index at the end.
    std::vector<std::string> args = {
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-xerror",
        "-protocol_whitelist", "file",
        "-format_whitelist", "wav,mp3,flac,mov,aac,ogg,matroska,webm,aiff,caf,asf,ape,amr,au,wv",
        "-threads", "1", "-i", input, "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-map_metadata", "-1", "-threads", "1", "-c:a", "pcm_f32le",
        "-fs", std::to_string(max_audio_bytes), "-f", "wav", output
    };
    std::vector<char *> argv;
    for (auto & arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        throw AudioDecodeError(500, "could not initialize audio decoder process");
    }
    // Do not let FFmpeg consume server stdin or emit arbitrary uploaded metadata
    // into the server log. Failures are reported as stable API errors below.
    int rc = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (!rc) rc = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    if (!rc) rc = posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = -1;
    if (!rc) rc = posix_spawnp(&pid, "ffmpeg", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) {
        throw AudioDecodeError(503, "audio decoder unavailable: install ffmpeg and make it available on the server PATH");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    int status = 0;
    while (true) {
        const auto result = waitpid(pid, &status, WNOHANG);
        if (result == pid) break;
        if (result < 0 && errno != EINTR) {
            throw AudioDecodeError(500, "could not wait for audio decoder");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            throw AudioDecodeError(400, "audio decoding exceeded the 60 second limit");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw AudioDecodeError(400, "could not decode uploaded audio: invalid, corrupt, or unsupported audio file");
    }
}
#endif
} // namespace

std::string decode_uploaded_audio(std::string_view bytes) {
    if (bytes.empty()) throw AudioDecodeError(400, "uploaded audio is empty");
    if (bytes.size() >= max_audio_bytes) throw AudioDecodeError(413, "uploaded audio exceeds the 256 MiB limit");
    if (native_wav(bytes)) return std::string(bytes);
#ifdef _WIN32
    throw AudioDecodeError(503, "this build supports native WAV uploads only; automatic FFmpeg decoding requires a POSIX server");
#else
    TemporaryAudio temp;
    const auto input = temp.directory / "input";
    const auto output = temp.directory / "output.wav";
    {
        std::ofstream file(input, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        file.close();
        if (!file) throw AudioDecodeError(500, "could not write temporary audio input");
    }
    run_ffmpeg(input.string(), output.string());
    std::error_code ec;
    const auto size = std::filesystem::file_size(output, ec);
    if (ec || size < 44) throw AudioDecodeError(400, "audio decoder produced no audio");
    // FFmpeg's -fs can stop successfully with truncated audio; never transcribe
    // such output as if it represented the complete upload.
    if (size >= max_audio_bytes) throw AudioDecodeError(413, "decoded audio exceeds the 256 MiB limit");
    std::string wav(static_cast<std::size_t>(size), '\0');
    std::ifstream file(output, std::ios::binary);
    if (!file.read(wav.data(), static_cast<std::streamsize>(wav.size()))) {
        throw AudioDecodeError(500, "could not read decoded audio");
    }
    if (!native_wav(wav)) throw AudioDecodeError(400, "audio decoder produced invalid WAV output");
    return wav;
#endif
}
} // namespace minitts::server
