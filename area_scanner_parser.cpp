#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

#define MMWDEMO_OUTPUT_MSG_SEGMENT_LEN 32

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

// Function to parse TLV data based on type
void parseTLV(uint8_t* payload, uint32_t type, uint32_t length) {
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

// Function to parse the AreaScannerFrame
void parseAreaScannerFrame(uint8_t* frameData, uint32_t frameLength) {
    MmwDemo_output_message_header_t* header = reinterpret_cast<MmwDemo_output_message_header_t*>(frameData);

    // Verify magic word for frame validity
    const uint16_t magicWordExpected[4] = {0x0102, 0x0304, 0x0506, 0x0708};
    if (memcmp(header->magicWord, magicWordExpected, sizeof(magicWordExpected)) != 0) {
        std::cerr << "Invalid magic word!" << std::endl;
        return;
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
}

int main() {
    // Example usage: Load a frame (dummy data for demonstration)
    uint8_t frameData[1024] = {0x02,0x01,0x04,0x03,0x06,0x05,0x08,0x07,0x04,0x00,0x05,0x03,0x80,0x00,0x00,0x00,0x43,0x68,0x0a,0x00,0x01,0x00,0x00,0x00,0x14,0xa8,0xab,0x4b,0x02,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x20,0x00,0x00,0x00,0xe2,0x1e,0x46,0x3f,0x20,0x44,0x91,0x3f,0xf7,0xe0,0xbb,0xbc,0x12,0xb8,0x16,0xbe,0xb1,0x21,0x58,0x3f,0x3d,0x4b,0x9c,0x3f,0x48,0x42,0x84,0x3d,0x12,0xb8,0x16,0xbe,0x07,0x00,0x00,0x00,0x08,0x00,0x00,0x00,0x9d,0x00,0x9d,0x01,0x96,0x00,0x85,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};  // Example data buffer
    uint32_t frameLength = 128;   // Example frame length

    // Call the parser
    parseAreaScannerFrame(frameData, frameLength);

    return 0;
}
