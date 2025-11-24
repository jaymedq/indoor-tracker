#include <gtest/gtest.h>
#include "mmWave/frames/AreaScannerFrame.hpp"

#include <gtest/gtest.h>
#include "mmWave/frames/AreaScannerFrame.hpp"

TEST(AreaScannerFrameTest, ParseAndDisplay) {
    // Create a mock data vector for an AreaScannerFrame
    MmwDemo_output_message_header_t header;
    header.magicWord[0] = 0x0102;
    header.magicWord[1] = 0x0304;
    header.magicWord[2] = 0x0506;
    header.magicWord[3] = 0x0708;
    header.version = 0x01020304;
    header.totalPacketLen = sizeof(header) + sizeof(MmwDemo_output_message_tl_t) + sizeof(DPIF_PointCloudSpherical_t);
    header.platform = 0x090A0B0C;
    header.frameNumber = 0x100F0E0D;
    header.timeCpuCycles = 0x14131211;
    header.numDetectedObj = 1;
    header.numTLVs = 1;
    header.subFrameNumber = 0x201F1E1D;

    MmwDemo_output_message_tl_t tlv;
    tlv.type = MMWDEMO_OUTPUT_MSG_DETECTED_POINTS;
    tlv.length = sizeof(DPIF_PointCloudSpherical_t);

    DPIF_PointCloudSpherical_t point;
    point.range = 1.0f;
    point.azimuthAngle = 0.5f;
    point.elevAngle = 0.2f;
    point.velocity = 0.1f;

    std::vector<uint8_t> mock_data;
    mock_data.resize(sizeof(header) + sizeof(tlv) + sizeof(point));
    memcpy(mock_data.data(), &header, sizeof(header));
    memcpy(mock_data.data() + sizeof(header), &tlv, sizeof(tlv));
    memcpy(mock_data.data() + sizeof(header) + sizeof(tlv), &point, sizeof(point));

    AreaScannerFrame frame;
    EXPECT_TRUE(frame.parse(mock_data));
    EXPECT_EQ(frame.getPointCloud().size(), 1);
    // We can't really test display() in a unit test, but we can call it to make sure it doesn't crash
    frame.display();
}
