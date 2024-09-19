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
    bool parse(const std::vector<uint8_t> &data) override;

    void display() const override;

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
};

#endif // AREA_SCANNER_FRAME_HPP