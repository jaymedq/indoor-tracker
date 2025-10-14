// parser.cpp
#include "parser.hpp"
#include <iostream>

// Factory function to create the appropriate frame
extern "C" __declspec(dllexport) std::unique_ptr<IFrame> createFrame(const std::vector<uint8_t> &data) {
    if (data.size() < 8) {
        std::cerr << "Invalid data size\n";
        return nullptr;
    }
    if (data[0] == 0x02 && data[1] == 0x01 && data[2] == 0x04 &&
        data[3] == 0x03) { // Check for AreaScannerFrame magic word
        return std::make_unique<AreaScannerFrame>();
    } else {
        std::cerr << "Unrecognized frame type\n";
        return nullptr;
    }
}

// Exported function for the parser
extern "C" __declspec(dllexport) int parse_frame(const uint8_t* data, size_t length) {
    std::vector<uint8_t> frameData(data, data + length);
    auto frame = createFrame(frameData);

    if (frame && frame->parse(frameData)) {
        frame->display();
        return 0;  // Success
    } else {
        std::cerr << "Failed to parse frame\n";
        return -1;  // Failure
    }
}
