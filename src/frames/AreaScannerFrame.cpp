/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */

#include "AreaScannerFrame.hpp"

bool AreaScannerFrame::parse(const std::vector<uint8_t> &data) override
{
    if (data.size() < 44)
    {
        std::cerr << "Invalid frame size.\n";
        return false;
    }
    // Magic Word detection (8 bytes)
    if (!checkMagicPattern(data.data()))
    {
        std::cerr << "Magic word not found!\n";
        return false;
    }
    // Parse fields according to the AreaScannerFrame structure
    version = getUint32(data, 8);
    totalPacketLength = getUint32(data, 12);
    platform = getUint32(data, 16);
    frameNumber = getUint32(data, 20);
    timeCpuCycles = getUint32(data, 24);
    numDetectedObj = getUint32(data, 28);
    numTlv = getUint32(data, 32);
    subFrameNumber = getUint32(data, 36);
    numStaticDetectedObj = getUint32(data, 40);
    return true;
}

void AreaScannerFrame::display() const override
{
    std::cout << "AreaScannerFrame\n";
    std::cout << "Version: " << version << "\n";
    std::cout << "Total Packet Length: " << totalPacketLength << "\n";
    std::cout << "Platform: " << platform << "\n";
    std::cout << "Frame Number: " << frameNumber << "\n";
    std::cout << "Time in CPU Cycles: " << timeCpuCycles << "\n";
    std::cout << "Num Detected Objects: " << numDetectedObj << "\n";
    std::cout << "Num TLVs: " << numTlv << "\n";
    std::cout << "Subframe Number: " << subFrameNumber << "\n";
    std::cout << "Num Static Detected Objects: " << numStaticDetectedObj
              << "\n";
}

bool AreaScannerFrame::checkMagicPattern(const uint8_t *data) const
{
    return (data[0] == 0x02 && data[1] == 0x01 && data[2] == 0x04 &&
            data[3] == 0x03 && data[4] == 0x06 && data[5] == 0x05 &&
            data[6] == 0x08 && data[7] == 0x07);
}

uint32_t AreaScannerFrame::getUint32(const std::vector<uint8_t> &data, int offset) const
{
    return (data[offset] + (data[offset + 1] << 8) + (data[offset + 2] << 16) +
            (data[offset + 3] << 24));
}
