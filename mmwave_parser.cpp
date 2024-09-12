#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <cstring>
#include <iomanip>

// Base Frame class
class Frame {
public:
    virtual bool parse(const std::vector<uint8_t>& data) = 0;
    virtual void display() const = 0;
    virtual ~Frame() = default;
};

// DemoFrame class
class DemoFrame : public Frame {
public:
    bool parse(const std::vector<uint8_t>& data) override {
        const int headerNumBytes = 40;
        const float PI = 3.14159265;

        int headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber;
        parserHelper(data, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber);

        if (headerStartIndex == -1) {
            std::cerr << "Frame Fail: cannot find the magic words.\n";
            return false;
        }

        int nextHeaderStartIndex = headerStartIndex + totalPacketNumBytes;

        if (headerStartIndex + totalPacketNumBytes > data.size()) {
            std::cerr << "Frame Fail: readNumBytes may not be long enough.\n";
            return false;
        } else if (nextHeaderStartIndex + 8 < data.size() && !checkMagicPattern(data.data() + nextHeaderStartIndex)) {
            std::cerr << "Frame Fail: incomplete packet.\n";
            return false;
        } else if (numDetObj <= 0) {
            std::cerr << "Frame Fail: numDetObj = " << numDetObj << "\n";
            return false;
        } else if (subFrameNumber > 3) {
            std::cerr << "Frame Fail: subFrameNumber = " << subFrameNumber << "\n";
            return false;
        }

        // TLV Parsing
        parseTLVs(data, headerStartIndex + headerNumBytes, numDetObj, totalPacketNumBytes);
        return true;
    }

    void display() const override {
        std::cout << "DemoFrame parsed successfully\n";
    }

private:
    // Helper methods (same as before)
    uint32_t getUint32(const uint8_t* data) const {
        return (data[0] + (data[1] << 8) + (data[2] << 16) + (data[3] << 24));
    }

    uint16_t getUint16(const uint8_t* data) const {
        return (data[0] + (data[1] << 8));
    }

    bool checkMagicPattern(const uint8_t* data) const {
        return (data[0] == 2 && data[1] == 1 && data[2] == 4 && data[3] == 3 &&
                data[4] == 6 && data[5] == 5 && data[6] == 8 && data[7] == 7);
    }

    void parserHelper(const std::vector<uint8_t>& data, int& headerStartIndex, 
                      int& totalPacketNumBytes, int& numDetObj, int& numTlv, int& subFrameNumber) const {
        headerStartIndex = -1;

        for (size_t index = 0; index < data.size(); ++index) {
            if (checkMagicPattern(data.data() + index)) {
                headerStartIndex = index;
                break;
            }
        }

        if (headerStartIndex == -1) {
            totalPacketNumBytes = numDetObj = numTlv = subFrameNumber = -1;
            return;
        }

        totalPacketNumBytes = getUint32(data.data() + headerStartIndex + 12);
        numDetObj = getUint32(data.data() + headerStartIndex + 28);
        numTlv = getUint32(data.data() + headerStartIndex + 32);
        subFrameNumber = getUint32(data.data() + headerStartIndex + 36);
    }

    void parseTLVs(const std::vector<uint8_t>& data, int tlvStart, int numDetObj, int totalPacketNumBytes) const {
        const float PI = 3.14159265;

        // First TLV
        uint32_t tlvType = getUint32(data.data() + tlvStart);
        uint32_t tlvLen = getUint32(data.data() + tlvStart + 4);
        int offset = 8;

        std::cout << "The 1st TLV\n";
        std::cout << "    type " << tlvType << "\n";
        std::cout << "    len " << tlvLen << " bytes\n";

        if (tlvType == 1 && tlvLen < totalPacketNumBytes) {
            for (int obj = 0; obj < numDetObj; ++obj) {
                float x = *reinterpret_cast<const float*>(data.data() + tlvStart + offset);
                float y = *reinterpret_cast<const float*>(data.data() + tlvStart + offset + 4);
                float z = *reinterpret_cast<const float*>(data.data() + tlvStart + offset + 8);
                float v = *reinterpret_cast<const float*>(data.data() + tlvStart + offset + 12);

                float compDetectedRange = std::sqrt((x * x) + (y * y) + (z * z));
                float detectedAzimuth = (y == 0) ? ((x >= 0) ? 90 : -90) : std::atan(x / y) * 180 / PI;
                float detectedElevAngle = (x == 0 && y == 0) ? ((z >= 0) ? 90 : -90) : std::atan(z / std::sqrt((x * x) + (y * y))) * 180 / PI;

                std::cout << "x = " << x << ", y = " << y << ", z = " << z << ", v = " << v << ", range = " << compDetectedRange
                            << ", azimuth = " << detectedAzimuth << ", elevAngle = " << detectedElevAngle << "\n";

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

        if (tlvType == 7) {
            for (int obj = 0; obj < numDetObj; ++obj) {
                uint16_t snr = getUint16(data.data() + tlvStart + offset);
                uint16_t noise = getUint16(data.data() + tlvStart + offset + 2);

                std::cout << "snr = " << snr << ", noise = " << noise << "\n";

                offset += 4;
            }
        }
    }
};

// AreaScannerFrame class (inherits from Frame)
class AreaScannerFrame : public Frame {
public:
    bool parse(const std::vector<uint8_t>& data) override {
        if (data.size() < 44) {
            std::cerr << "Invalid frame size.\n";
            return false;
        }
        // Magic Word detection (8 bytes)
        if (!checkMagicPattern(data.data())) {
            std::cerr << "Magic word not found!\n";
            return false;
        }
        // Parse fields according to the AreaScannerFrame structure
        version = getUint32(data, 8);
        totalPacketLength = getUint32(data, 12);
        platform = getUint32(data, 16);
        frameNumber = getUint32(data, 20);
        timeCpuCycles = getUint32(data, 24);
        numDetectedObj = getUint32(data, 28);
        numTlv = getUint32(data, 32);
        subFrameNumber = getUint32(data, 36);
        numStaticDetectedObj = getUint32(data, 40);
        return true;
    }

    void display() const override {
        std::cout << "AreaScannerFrame\n";
        std::cout << "Version: " << version << "\n";
        std::cout << "Total Packet Length: " << totalPacketLength << "\n";
        std::cout << "Platform: " << platform << "\n";
        std::cout << "Frame Number: " << frameNumber << "\n";
        std::cout << "Time in CPU Cycles: " << timeCpuCycles << "\n";
        std::cout << "Num Detected Objects: " << numDetectedObj << "\n";
        std::cout << "Num TLVs: " << numTlv << "\n";
        std::cout << "Subframe Number: " << subFrameNumber << "\n";
        std::cout << "Num Static Detected Objects: " << numStaticDetectedObj << "\n";
    }

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

    uint32_t getUint32(const std::vector<uint8_t>& data, size_t index) {
        return (data[index] + (data[index + 1] << 8) + (data[index + 2] << 16) + (data[index + 3] << 24));
    }

    bool checkMagicPattern(const uint8_t* data) {
        return (data[0] == 2 && data[1] == 1 && data[2] == 4 && data[3] == 3 &&
                data[4] == 6 && data[5] == 5 && data[6] == 8 && data[7] == 7);
    }
};

// FrameFactory class
class FrameFactory {
public:
    static std::unique_ptr<Frame> createFrame(const std::vector<uint8_t>& data) {
        // Implement frame detection logic here
        if (isDemoFrame(data)) {
            return std::make_unique<DemoFrame>();
        } else if (isAreaScannerFrame(data)) {
            return std::make_unique<AreaScannerFrame>();
        } else {
            std::cerr << "Unknown frame type!\n";
            return nullptr;
        }
    }

private:
    // Implement detection logic to determine frame type
    static bool isDemoFrame(const std::vector<uint8_t>& data) {
        // Detect if it's a DemoFrame by checking some pattern
        return data.size() > 40 && data[0] == 2 && data[1] == 1; // Placeholder logic
    }

    static bool isAreaScannerFrame(const std::vector<uint8_t>& data) {
        // Detect if it's an AreaScannerFrame by checking some pattern
        return data.size() > 30 && data[0] == 0xAA && data[1] == 0xBB; // Placeholder logic
    }
};

// Main function
int main() {
    
    std::vector<uint8_t> data;
    // Read hex byte stream from input (without space separation)
    std::string inputLine;
    std::cout << "Enter hex stream (without spaces):\n";
    std::getline(std::cin, inputLine);

    // Check if the input length is even
    if (inputLine.length() % 2 != 0) {
        std::cerr << "Invalid input length. Hex stream should have an even number of characters.\n";
        return 1;
    }

    // Process the input string in pairs of two characters
    for (size_t i = 0; i < inputLine.length(); i += 2) {
        std::string hexByte = inputLine.substr(i, 2);
        
        // Convert hex string to uint8_t
        uint8_t byte = static_cast<uint8_t>(std::stoul(hexByte, nullptr, 16));
        data.push_back(byte);
    }

    // Create the appropriate frame using the factory
    std::unique_ptr<Frame> frame = FrameFactory::createFrame(data);
    
    if (frame && frame->parse(data)) {
        frame->display();
    } else {
        std::cerr << "Parsing failed.\n";
    }

    return 0;
}