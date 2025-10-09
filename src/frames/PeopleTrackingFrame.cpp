/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */
#include "PeopleTrackingFrame.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>

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
        std::vector<uint8_t> tlvSlice(data.begin() + currentOffset, data.begin() + currentOffset + tl.length);
        if(tl.length > tlvSlice.size()) {
            std::cerr << "TLV length exceeds available data!" << std::endl;
            return false;
        }
        if(tl.length > 0)
        {
            parseTLV(tlvSlice, tl.type, tl.length);
        }
        currentOffset += tl.length;
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
            m_compressedPointCloud = {};
            m_compressedPointCloud.header.type = type;
            m_compressedPointCloud.header.length = length;
            memcpy(&m_compressedPointCloud.pointUnit, payload.data(), sizeof(m_compressedPointCloud.pointUnit));
            m_u32NumPoints = (length - sizeof(MmwDemo_output_message_compressedPoint_unit)) / sizeof(MmwDemo_output_message_compressedPoint);
            for(uint32_t i = 0; i < m_u32NumPoints; ++i) {
                MmwDemo_output_message_compressedPoint point = {};
                memcpy(&point,  payload.data() + sizeof(MmwDemo_output_message_compressedPoint_unit) + (1+i) * sizeof(MmwDemo_output_message_compressedPoint), sizeof(MmwDemo_output_message_compressedPoint));
                m_compressedPointCloud.pointCloud.push_back(point);
            }
            std::cout << "Point Cloud TLV: " << m_compressedPointCloud.header.length << " bytes, Points: " << m_u32NumPoints << std::endl;
            break;
        }
        case PRESENCE_INDICATION:
        {
            memcpy(&m_presenceIndication, payload.data(), sizeof(m_presenceIndication));
            break;
        }
        case TARGET_LIST:
        {            
            uint16_t numTargets = length / sizeof(trackerProc_Target);
            for(uint32_t i = 0; i < numTargets; i++)
            {
                trackerProc_Target target = {};
                memcpy(&target, payload.data() + i * sizeof(trackerProc_Target), sizeof(trackerProc_Target));
                m_targets.push_back(target);
            }
            break;
        }
        case TARGET_INDEX:
        {
            if(length != sizeof(m_targetID)) {
                std::cerr << "Unexpected TARGET_INDEX length: " << length << std::endl;
                break;
            }
            memcpy(&m_targetID, payload.data(), sizeof(m_targetID));
            break;
        }
        case TARGET_HEIGHT:
        {
            uint32_t numHeights = length / sizeof(float);
            for(uint32_t i = 0; i < numHeights; i++)
            {
                heightDet_TargetHeight height = {};
                memcpy(&height, payload.data() + i * sizeof(heightDet_TargetHeight), sizeof(heightDet_TargetHeight));
                m_targetHeights.push_back(height);
            }
            break;
        }
        default:
        {            
            std::cout << "Unknown TLV type: " << type << " with length: " << length << std::endl;
            break;
        }
    }
}

void PeopleTrackingFrame::display() const
{
    std::cout << "PeopleTrackingFrame parsed successfully\n";
}
void PeopleTrackingFrame::toCsv(const std::string &path) const
{
    // Open the file in append mode
    std::ofstream file;
    file.open(path, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << path << std::endl;
        return;
    }

    // Get the current time and format it as dd/MM/yyyy HH:mm
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *local_time = std::localtime(&now_time);

    std::ostringstream timeStream;
    timeStream << std::put_time(local_time, "%d/%m/%Y %H:%M:%S");
    file << m_compressedPointCloud.pointCloud.size() << ",";

    std::vector<DPIF_PointCloudCartesian_t> cartesianPointCloud;
    for (size_t i = 0; i < m_compressedPointCloud.pointCloud.size(); ++i) {
        DPIF_PointCloudCartesian_t cartesianPoint;
        MmwDemo_output_message_compressedPoint detectedPoint = m_compressedPointCloud.pointCloud[i];
        detectedPoint.azimuth *= m_compressedPointCloud.pointUnit.azimuthUnit;
        detectedPoint.elevation *= m_compressedPointCloud.pointUnit.elevationUnit;
        detectedPoint.doppler *= m_compressedPointCloud.pointUnit.dopplerUnit;
        detectedPoint.range *= m_compressedPointCloud.pointUnit.rangeUnit;
        detectedPoint.snr *= m_compressedPointCloud.pointUnit.snrUint;
        // Convert degrees to radians for trigonometric functions
        float elev_radians = detectedPoint.elevation * (M_PI / 180.0);
        float azimuth_radians = detectedPoint.azimuth * (M_PI / 180.0);
        cartesianPoint.x = detectedPoint.range * cos(elev_radians) * sin(azimuth_radians);
        cartesianPoint.y = detectedPoint.range * cos(elev_radians) * cos(azimuth_radians);
        cartesianPoint.z = detectedPoint.range * sin(elev_radians);
        cartesianPoint.velocity = detectedPoint.doppler;
        cartesianPointCloud.push_back(cartesianPoint);
    }

    file << "\"[";
    for (size_t i = 0; i < cartesianPointCloud.size(); ++i) {
        file << cartesianPointCloud[i].x;
        if (i < cartesianPointCloud.size() - 1) {
            file << ", ";
        }
    }
    file << "]\",";
    file << "\"[";
    for (size_t i = 0; i < cartesianPointCloud.size(); ++i) {
        file << cartesianPointCloud[i].y;
        if (i < cartesianPointCloud.size() - 1) {
            file << ", ";
        }
    }
    file << "]\",";
    file << "\"[";
    for (size_t i = 0; i < cartesianPointCloud.size(); ++i) {
        file << cartesianPointCloud[i].z;
        if (i < cartesianPointCloud.size() - 1) {
            file << ", ";
        }
    }
    file << "]\",";
    file << "\"[";
    for (size_t i = 0; i < cartesianPointCloud.size(); ++i) {
        file << cartesianPointCloud[i].velocity;
        if (i < cartesianPointCloud.size() - 1) {
            file << ", ";
        }
    }
    file << "]\",";
    file << timeStream.str() << "\n";
    file.close();
    std::cout << "Point cloud data has been written to: " << path << std::endl;
}
