#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace minitts::server {

class AudioDecodeError : public std::runtime_error {
public:
    AudioDecodeError(int status, const std::string & message)
        : std::runtime_error(message), status(status) {}
    int status;
};

// Keep PCM WAV uploads in memory; decode other containers using FFmpeg on POSIX.
// Neither the client filename nor its MIME type is trusted for format detection.
std::string decode_uploaded_audio(std::string_view bytes);

} // namespace minitts::server
