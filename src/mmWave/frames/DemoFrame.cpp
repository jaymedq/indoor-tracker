/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */

#include "DemoFrame.hpp"
#include <iostream>
#include <cmath>

bool DemoFrame::parse(std::vector<uint8_t> &data)
{
    const int headerNumBytes = 40;
    const float PI = 3.14159265;

    int headerStartIndex, totalPacketNumBytes, numDetObj, numTlv,
        subFrameNumber;
    parserHelper(data, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv,
                 subFrameNumber);

    if (headerStartIndex == -1)
    {
        std::cerr << "Frame Fail: cannot find the magic words.\n";
        return false;
    }

    int nextHeaderStartIndex = headerStartIndex + totalPacketNumBytes;

    if (headerStartIndex + totalPacketNumBytes > data.size())
    {
        std::cerr << "Frame Fail: readNumBytes may not be long enough.\n";
        return false;
    }
    else if (nextHeaderStartIndex + 8 < data.size() &&
             !checkMagicPattern(data.data() + nextHeaderStartIndex))
    {
        std::cerr << "Frame Fail: incomplete packet.\n";
        return false;
    }
    else if (numDetObj <= 0)
    {
        std::cerr << "Frame Fail: numDetObj = " << numDetObj << "\n";
        return false;
    }
    else if (subFrameNumber > 3)
    {
        std::cerr << "Frame Fail: subFrameNumber = " << subFrameNumber << "\n";
        return false;
    }

    // TLV Parsing
    parseTLVs(data, headerStartIndex + headerNumBytes, numDetObj,
              totalPacketNumBytes);
    return true;
}

void DemoFrame::display() const
{
    std::cout << "DemoFrame parsed successfully\n";
}

// Helper methods (same as before)
uint32_t DemoFrame::getUint32(const uint8_t *data) const
{
    return (data[0] + (data[1] << 8) + (data[2] << 16) + (data[3] << 24));
}

uint16_t DemoFrame::getUint16(const uint8_t *data) const
{
    return (data[0] + (data[1] << 8));
}

bool DemoFrame::checkMagicPattern(const uint8_t *data) const
{
    return (data[0] == 2 && data[1] == 1 && data[2] == 4 && data[3] == 3 &&
            data[4] == 6 && data[5] == 5 && data[6] == 8 && data[7] == 7);
}

void DemoFrame::parserHelper(const std::vector<uint8_t> &data, int &headerStartIndex,
                  int &totalPacketNumBytes, int &numDetObj, int &numTlv,
                  int &subFrameNumber) const
{
    headerStartIndex = -1;

    for (size_t index = 0; index < data.size(); ++index)
    {
        if (checkMagicPattern(data.data() + index))
        {
            headerStartIndex = index;
            break;
        }
    }

    if (headerStartIndex == -1)
    {
        totalPacketNumBytes = numDetObj = numTlv = subFrameNumber = -1;
        return;
    }

    totalPacketNumBytes = getUint32(data.data() + headerStartIndex + 12);
    numDetObj = getUint32(data.data() + headerStartIndex + 28);
    numTlv = getUint32(data.data() + headerStartIndex + 32);
    subFrameNumber = getUint32(data.data() + headerStartIndex + 36);
}

void DemoFrame::parseTLVs(const std::vector<uint8_t> &data, int tlvStart, int numDetObj,
               int totalPacketNumBytes) const
{
    const float PI = 3.14159265;

    // First TLV
    uint32_t tlvType = getUint32(data.data() + tlvStart);
    uint32_t tlvLen = getUint32(data.data() + tlvStart + 4);
    int offset = 8;

    std::cout << "The 1st TLV\n";
    std::cout << "    type " << tlvType << "\n";
    std::cout << "    len " << tlvLen << " bytes\n";

    if (tlvType == 1 && tlvLen < totalPacketNumBytes)
    {
        for (int obj = 0; obj < numDetObj; ++obj)
        {
            float x =
                *reinterpret_cast<const float *>(data.data() + tlvStart + offset);
            float y = *reinterpret_cast<const float *>(data.data() + tlvStart +
                                                       offset + 4);
            float z = *reinterpret_cast<const float *>(data.data() + tlvStart +
                                                       offset + 8);
            float v = *reinterpret_cast<const float *>(data.data() + tlvStart +
                                                       offset + 12);

            float compDetectedRange = std::sqrt((x * x) + (y * y) + (z * z));
            float detectedAzimuth =
                (y == 0) ? ((x >= 0) ? 90 : -90) : std::atan(x / y) * 180 / PI;
            float detectedElevAngle =
                (x == 0 && y == 0)
                    ? ((z >= 0) ? 90 : -90)
                    : std::atan(z / std::sqrt((x * x) + (y * y))) * 180 / PI;

            std::cout << "x = " << x << ", y = " << y << ", z = " << z
                      << ", v = " << v << ", range = " << compDetectedRange
                      << ", azimuth = " << detectedAzimuth
                      << ", elevAngle = " << detectedElevAngle << "\n";

            offset += 16;
        }
    }

    // Second TLV
    tlvStart = tlvStart + 8 + tlvLen;
    tlvType = getUint32(data.data() + tlvStart);
    tlvLen = getUint32(data.data() + tlvStart + 4);
    offset = 8;

    std::cout << "The 2nd TLV\n";
    std::cout << "    type " << tlvType << "\n";
    std::cout << "    len " << tlvLen << " bytes\n";

    if (tlvType == 7)
    {
        for (int obj = 0; obj < numDetObj; ++obj)
        {
            uint16_t snr = getUint16(data.data() + tlvStart + offset);
            uint16_t noise = getUint16(data.data() + tlvStart + offset + 2);

            std::cout << "snr = " << snr << ", noise = " << noise << "\n";

            offset += 4;
        }
    }
}