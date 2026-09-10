#include "audio_decode.h"
#include "engine/framework/audio/wav_reader.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using minitts::server::AudioDecodeError;
using minitts::server::decode_uploaded_audio;

int main(int argc, char ** argv) {
    try {
        if (argc == 3) {
            std::ifstream input(argv[1], std::ios::binary);
            const std::string bytes((std::istreambuf_iterator<char>(input)), {});
            const auto wav = decode_uploaded_audio(bytes);
            const auto audio = engine::audio::read_wav_f32(std::string_view(wav));
            if (audio.samples.empty()) throw std::runtime_error("native reader produced no samples");
            std::ofstream output(argv[2], std::ios::binary);
            output.write(wav.data(), wav.size());
            if (!output) throw std::runtime_error("could not write test output");
            return 0;
        }
        try {
            decode_uploaded_audio("");
            throw std::runtime_error("empty input accepted");
        } catch (const AudioDecodeError & ex) {
            if (ex.status != 400) throw;
        }
#ifndef _WIN32
        // A valid PCM WAV must remain usable even when FFmpeg is absent.
        const char * path = std::getenv("PATH");
        const std::string saved_path = path ? path : "";
        setenv("PATH", "/nonexistent-audiocpp-test-path", 1);
#endif
        const unsigned char raw[] = {
            'R','I','F','F',38,0,0,0,'W','A','V','E',
            'f','m','t',' ',16,0,0,0,1,0,1,0,0x80,0x3e,0,0,
            0,0x7d,0,0,2,0,16,0,'d','a','t','a',2,0,0,0,0,0
        };
        const std::string wav(reinterpret_cast<const char *>(raw), sizeof(raw));
        if (decode_uploaded_audio(wav) != wav) throw std::runtime_error("native WAV changed");
        try {
            decode_uploaded_audio("not an audio file");
            throw std::runtime_error("missing decoder accepted input");
        } catch (const AudioDecodeError & ex) {
            if (ex.status != 503) throw;
        }
#ifndef _WIN32
        if (path) setenv("PATH", saved_path.c_str(), 1);
        else unsetenv("PATH");
#endif
        std::cout << "audio decoding unit tests passed\n";
        return 0;
    } catch (const AudioDecodeError & ex) {
        std::cerr << ex.status << ": " << ex.what() << '\n';
        return 1;
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
