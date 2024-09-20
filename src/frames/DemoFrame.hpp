/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 */

#if !defined(DEMO_FRAME_HPP)
#define DEMO_FRAME_HPP

#include "IFrame.hpp"

class DemoFrame : public IFrame
{
public:
    bool parse(std::vector<uint8_t> &data) override;

    void display() const override;

private:
    // Helper methods (same as before)
    uint32_t getUint32(const uint8_t *data) const;

    uint16_t getUint16(const uint8_t *data) const;

    bool checkMagicPattern(const uint8_t *data) const;

    void parserHelper(const std::vector<uint8_t> &data, int &headerStartIndex,
                      int &totalPacketNumBytes, int &numDetObj, int &numTlv,
                      int &subFrameNumber) const;

    void parseTLVs(const std::vector<uint8_t> &data, int tlvStart, int numDetObj,
                   int totalPacketNumBytes) const;
};

#endif // DEMO_FRAME_HPP