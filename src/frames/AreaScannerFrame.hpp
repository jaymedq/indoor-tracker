/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */
#if !defined(AREA_SCANNER_FRAME_HPP)
#define AREA_SCANNER_FRAME_HPP

#include "IFrame.hpp"

// AreaScannerFrame class (inherits from Frame)
class AreaScannerFrame : public IFrame
{
public:
    bool parse(std::vector<uint8_t> &data) override;

    void display() const override;

    // Define message types as provided
    enum MmwDemo_output_message_type_e {
        MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1,
        MMWDEMO_OUTPUT_MSG_RANGE_PROFILE,
        MMWDEMO_OUTPUT_MSG_NOISE_PROFILE,
        MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP,
        MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP,
        MMWDEMO_OUTPUT_MSG_STATS,
        MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO,
        MMWDEMO_OUTPUT_MSG_STATIC_DETECTED_POINTS,
        MMWDEMO_OUTPUT_MSG_STATIC_DETECTED_POINTS_SIDE_INFO,
        MMWDEMO_OUTPUT_MSG_TRACKERPROC_OUTPUT_TARGET_LIST,
        MMWDEMO_OUTPUT_MSG_TRACKERPROC_OUTPUT_TARGET_INDEX,
        MMWDEMO_OUTPUT_MSG_MAX
    };

    // Message header structure
    struct MmwDemo_output_message_header_t {
        uint16_t magicWord[4];
        uint32_t version;
        uint32_t totalPacketLen;
        uint32_t platform;
        uint32_t frameNumber;
        uint32_t timeCpuCycles;
        uint32_t numDetectedObj;
        uint32_t numTLVs;
        uint32_t subFrameNumber;
        uint32_t numStaticDetectedObj;
    };

    // TLV header structure
    struct MmwDemo_output_message_tl_t {
        uint32_t type;
        uint32_t length;
    };

    // Detected points structure (Example based on `DPIF_PointCloudCartesian_t`)
    struct DPIF_PointCloudCartesian_t {
        float x;
        float y;
        float z;
        float velocity;
    };

    // Stats structure
    struct MmwDemo_output_message_stats_t {
        uint32_t interFrameProcessingTime;
        uint32_t transmitOutputTime;
        uint32_t interFrameProcessingMargin;
        uint32_t interChirpProcessingMargin;
        uint32_t activeFrameCPULoad;
        uint32_t interFrameCPULoad;
    };

private:
    uint32_t version;
    uint32_t totalPacketLength;
    uint32_t platform;
    uint32_t frameNumber;
    uint32_t timeCpuCycles;
    uint32_t numDetectedObj;
    uint32_t numTlv;
    uint32_t subFrameNumber;
    uint32_t numStaticDetectedObj;

    bool checkMagicPattern(const uint8_t *data) const;
    uint32_t getUint32(const std::vector<uint8_t> &data, int offset) const;
    void parseTLV(uint8_t* payload, uint32_t type, uint32_t length);
};

#endif // AREA_SCANNER_FRAME_HPP