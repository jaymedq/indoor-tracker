/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */

#include "AreaScannerFrame.hpp"
#include <iostream>
#include <cstring>

bool AreaScannerFrame::parse(std::vector<uint8_t> &data)
{
    uint8_t* frameData = data.data();
    uint32_t frameLength = data.size();
    MmwDemo_output_message_header_t* header = reinterpret_cast<MmwDemo_output_message_header_t*>(frameData);

    // Verify magic word for frame validity
    const uint16_t magicWordExpected[4] = {0x0102, 0x0304, 0x0506, 0x0708};
    if (std::memcmp(header->magicWord, magicWordExpected, sizeof(magicWordExpected)) != 0) {
        std::cerr << "Invalid magic word!" << std::endl;
        return false;
    }

    std::cout << "Frame Number: " << header->frameNumber << std::endl;
    std::cout << "Detected Objects: " << header->numDetectedObj << std::endl;
    std::cout << "Number of TLVs: " << header->numTLVs << std::endl;

    // Iterate over TLVs
    uint8_t* tlvPtr = frameData + sizeof(MmwDemo_output_message_header_t);
    for (uint32_t i = 0; i < header->numTLVs; ++i) {
        MmwDemo_output_message_tl_t* tlv = reinterpret_cast<MmwDemo_output_message_tl_t*>(tlvPtr);

        // Parse the payload associated with the TLV
        uint8_t* payload = tlvPtr + sizeof(MmwDemo_output_message_tl_t);
        parseTLV(payload, tlv->type, tlv->length);

        // Move to the next TLV (type + length + payload size)
        tlvPtr += sizeof(MmwDemo_output_message_tl_t) + tlv->length;
    }
    return true;
}

void AreaScannerFrame::display() const
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

// Function to parse TLV data based on type
void AreaScannerFrame::parseTLV(uint8_t* payload, uint32_t type, uint32_t length) {
    switch (type) {
        case MMWDEMO_OUTPUT_MSG_DETECTED_POINTS: {
            int numDetectedPoints = length / sizeof(DPIF_PointCloudCartesian_t);
            DPIF_PointCloudCartesian_t* detectedPoints = reinterpret_cast<DPIF_PointCloudCartesian_t*>(payload);

            std::cout << "Detected Points: " << numDetectedPoints << std::endl;
            for (int i = 0; i < numDetectedPoints; ++i) {
                std::cout << "Point " << i << ": (" << detectedPoints[i].x << ", " 
                          << detectedPoints[i].y << ", " << detectedPoints[i].z 
                          << "), Velocity: " << detectedPoints[i].velocity << std::endl;
            }
            break;
        }
        case MMWDEMO_OUTPUT_MSG_STATS: {
            MmwDemo_output_message_stats_t* stats = reinterpret_cast<MmwDemo_output_message_stats_t*>(payload);
            std::cout << "Stats: InterFrameProcessingTime: " << stats->interFrameProcessingTime 
                      << ", TransmitOutputTime: " << stats->transmitOutputTime << std::endl;
            break;
        }
        // Add cases for other message types as needed
        default:
            std::cout << "Unknown TLV type: " << type << std::endl;
            break;
    }
}