/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */

#include "AreaScannerFrame.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <cmath>

bool AreaScannerFrame::parse(std::vector<uint8_t> &inputData) {
    uint32_t currentOffset = 0ULL;

    memcpy(&header,  &inputData.at(currentOffset), sizeof(header));
    currentOffset += sizeof(header);

    // Verify magic word for frame validity
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
        memcpy(&tl,  &inputData.at(currentOffset), sizeof(tl));
        currentOffset += sizeof(tl);

        // Parse the payload associated with the TLV
        std::vector<uint8_t> tlvSlice(inputData.begin() + currentOffset, inputData.end());
        parseTLV(tlvSlice, tl.type, tl.length);
    }
    return true;
}

void AreaScannerFrame::display() const
{
    std::cout << "AreaScannerFrame\n";
    std::cout << "Version: " << header.version << "\n";
    std::cout << "Total Packet Length: " << header.totalPacketLen << "\n";
    std::cout << "Platform: " << header.platform << "\n";
    std::cout << "Frame Number: " << header.frameNumber << "\n";
    std::cout << "Time in CPU Cycles: " << header.timeCpuCycles << "\n";
    std::cout << "Num Detected Objects: " << header.numDetectedObj << "\n";
    std::cout << "Num TLVs: " << header.numTLVs << "\n";
    std::cout << "Subframe Number: " << header.subFrameNumber << "\n";
    std::cout << "Num Static Detected Objects: " << header.numStaticDetectedObj
              << "\n";
}

void AreaScannerFrame::toCsv(const std::string& filePath) const {

    if(pointCloud.size() < 1 && staticObjectsPointCloud.size() < 1)
    {
        std::cout << "No pointcloud to be written to: " << filePath << std::endl;
    }
    else
    {
        // Open the file in append mode
        std::ofstream file;
        file.open(filePath, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file: " << filePath << std::endl;
            return;
        }

        // Get the current time and format it as dd/MM/yyyy HH:mm
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm *local_time = std::localtime(&now_time);

        std::ostringstream timeStream;
        timeStream << std::put_time(local_time, "%d/%m/%Y %H:%M:%S");
        file << pointCloud.size() << ",";
        file << "\"[";
        for (size_t i = 0; i < pointCloud.size(); ++i) {
            file << pointCloud[i].x;
            if (i < pointCloud.size() - 1) {
                file << ", ";
            }
        }
        file << "]\",";
        file << "\"[";
        for (size_t i = 0; i < pointCloud.size(); ++i) {
            file << pointCloud[i].y;
            if (i < pointCloud.size() - 1) {
                file << ", ";
            }
        }
        file << "]\",";
        file << "\"[";
        for (size_t i = 0; i < pointCloud.size(); ++i) {
            file << pointCloud[i].z;
            if (i < pointCloud.size() - 1) {
                file << ", ";
            }
        }
        file << "]\",";
        file << "\"[";
        for (size_t i = 0; i < pointCloud.size(); ++i) {
            file << pointCloud[i].velocity;
            if (i < pointCloud.size() - 1) {
                file << ", ";
            }
        }
        file << "]\",";
        file << timeStream.str() << "\n";
        file.close();
        std::cout << "Point cloud data has been written to: " << filePath << std::endl;
    }
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
void AreaScannerFrame::parseTLV(std::vector<uint8_t> payload, uint32_t type, uint32_t length) {
    switch (type) {
        case MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:
        {
            if (length % sizeof(DPIF_PointCloudSpherical_t) != 0) {
                std::cerr << "Payload size mismatch for detected points. Length: " << length 
                          << ", expected multiple of " << sizeof(DPIF_PointCloudSpherical_t) << std::endl;
                return;
            }
            uint16_t numDetectedPoints = length / sizeof(DPIF_PointCloudSpherical_t);
            std::vector<DPIF_PointCloudSpherical_t> detectedPoints(numDetectedPoints);
            std::memcpy(detectedPoints.data(), payload.data(), length);

            std::cout << "Detected Points: " << numDetectedPoints << std::endl;
            for (int i = 0; i < detectedPoints.size(); ++i) {
                DPIF_PointCloudCartesian_t cartesianPoint;
                const float elev_radians = detectedPoints[i].elevAngle;
                const float azimuth_radians = detectedPoints[i].azimuthAngle;
                const float range = detectedPoints[i].range;
                cartesianPoint.x = range * cos(elev_radians) * cos(azimuth_radians);
                cartesianPoint.y = range * cos(elev_radians) * sin(azimuth_radians);
                cartesianPoint.z = range * sin(elev_radians);
                cartesianPoint.velocity = detectedPoints[i].velocity;
                pointCloud.push_back(cartesianPoint);
                std::cout << "Point " << i << ": (" << cartesianPoint.x << ", " 
                            << cartesianPoint.y << ", " << cartesianPoint.z 
                            << "), Velocity: " << cartesianPoint.velocity << std::endl;
            }
            break;
        }
        case MMWDEMO_OUTPUT_MSG_STATS: {
            if (payload.size() < sizeof(stats)) {
                std::cerr << "Payload size mismatch for stats. Expected: " << sizeof(stats)
                          << ", got: " << payload.size() << std::endl;
                return;
            }
            std::memcpy(&stats, payload.data(), sizeof(stats));
            std::cout << "Stats: InterFrameProcessingTime: " << stats.interFrameProcessingTime 
                      << ", TransmitOutputTime: " << stats.transmitOutputTime << std::endl;
            break;
        }
        default:
            std::cout << "Unknown TLV type: " << type << std::endl;
            break;
    }
}