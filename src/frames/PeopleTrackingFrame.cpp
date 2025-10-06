/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */
#include "PeopleTrackingFrame.hpp"
#include <iostream>
#include <cstring>

bool PeopleTrackingFrame::parse(std::vector<uint8_t> &data)
{
    uint32_t currentOffset = 0ULL;
    PeopleTrackingFrame::MmwDemo_output_message_header_t header;
    memcpy(&header,  &data.at(currentOffset), sizeof(header));
    currentOffset += sizeof(header);

    const uint16_t magicWordExpected[4] = {0x0102, 0x0304, 0x0506, 0x0708};
    if (std::memcmp(header.magicWord, magicWordExpected, sizeof(magicWordExpected)) != 0) {
        std::cerr << "Invalid magic word!" << std::endl;
        return false;
    }
    
    std::cout << "Frame Number: " << header.frameNumber << std::endl;
    std::cout << "Detected Objects: " << header.numDetectedObj << std::endl;
    std::cout << "Number of TLVs: " << header.numTLVs << std::endl;

    // Iterate over TLVs
    for (uint32_t i = 0; i < header.numTLVs; ++i) {
        MmwDemo_output_message_tl_t tl = {};
        memcpy(&tl,  &data.at(currentOffset), sizeof(tl));
        currentOffset += sizeof(tl);

        // Parse the payload associated with the TLV
        std::vector<uint8_t> tlvSlice(data.begin() + currentOffset, data.end());
        if(tl.length > tlvSlice.size()) {
            std::cerr << "TLV length exceeds available data!" << std::endl;
            return false;
        }
        if(tl.length > 0)
        {
            parseTLV(tlvSlice, tl.type, tl.length);
        }
    }
    return true;
}

void PeopleTrackingFrame::parseTLV(std::vector<uint8_t> payload, uint32_t type, uint32_t length)
{
    static const uint32_t POINT_CLOUD = 1020;
    static const uint32_t TARGET_LIST = 1010;
    static const uint32_t TARGET_INDEX = 1011;
    static const uint32_t PRESENCE_INDICATION = 1021;
    static const uint32_t TARGET_HEIGHT = 1012;

    switch (type)
    {
        case POINT_CLOUD:
        {
            PeopleTrackingFrame::MmwDemo_output_message_compressedPointCloud_uart pointCloud = {};
            pointCloud.header.type = type;
            pointCloud.header.length = length;
            pointCloud.pointUint = {};
            memcpy(&pointCloud.pointUint, payload.data(), sizeof(pointCloud.pointUint));
            uint32_t numPoints = (length - sizeof(MmwDemo_output_message_compressedPoint_unit)) / sizeof(MmwDemo_output_message_compressedPoint);
            memcpy(&pointCloud.point,  payload.data() + sizeof(MmwDemo_output_message_compressedPoint_unit), numPoints * sizeof(MmwDemo_output_message_compressedPoint));
            std::cout << "Point Cloud TLV: " << pointCloud.header.length << " bytes, Points: " << length / sizeof(MmwDemo_output_message_compressedPoint) << std::endl;
            break;
        }
        default:
            std::cout << "Unknown TLV type: " << type << " with length: " << length << std::endl;
            break;
    }
}

void PeopleTrackingFrame::display() const
{
    std::cout << "PeopleTrackingFrame parsed successfully\n";
}
void PeopleTrackingFrame::toCsv(const std::string &path) const
{
    std::cout << "Exporting PeopleTrackingFrame data to CSV at: " << path << std::endl;
}
