#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TC_PASS 0
#define TC_FAIL 1

uint32_t getUint32(const uint8_t* data) {
    return (data[0] +
            (data[1] << 8) +
            (data[2] << 16) +
            (data[3] << 24));
}

uint16_t getUint16(const uint8_t* data) {
    return (data[0] +
            (data[1] << 8));
}

void getHex(const uint8_t* data, char* output) {
    sprintf(output, "%02x%02x%02x%02x", data[3], data[2], data[1], data[0]);
}

int checkMagicPattern(const uint8_t* data) {
    return (data[0] == 2 && data[1] == 1 && data[2] == 4 && data[3] == 3 &&
            data[4] == 6 && data[5] == 5 && data[6] == 8 && data[7] == 7);
}

void parser_helper(const uint8_t* data, int readNumBytes, int* headerStartIndex, 
                   int* totalPacketNumBytes, int* numDetObj, int* numTlv, int* subFrameNumber) {
    *headerStartIndex = -1;

    for (int index = 0; index < readNumBytes; index++) {
        if (checkMagicPattern(data + index)) {
            *headerStartIndex = index;
            break;
        }
    }

    if (*headerStartIndex == -1) { 
        *totalPacketNumBytes = -1;
        *numDetObj = -1;
        *numTlv = -1;
        *subFrameNumber = -1;
    } else {
        *totalPacketNumBytes = getUint32(data + *headerStartIndex + 12);
        char platform[9];
        getHex(data + *headerStartIndex + 16, platform);
        uint32_t frameNumber = getUint32(data + *headerStartIndex + 20);
        uint32_t timeCpuCycles = getUint32(data + *headerStartIndex + 24);
        *numDetObj = getUint32(data + *headerStartIndex + 28);
        *numTlv = getUint32(data + *headerStartIndex + 32);
        *subFrameNumber = getUint32(data + *headerStartIndex + 36);

        printf("headerStartIndex    = %d\n", *headerStartIndex);
        printf("totalPacketNumBytes = %d\n", *totalPacketNumBytes);
        printf("platform            = %s\n", platform);
        printf("frameNumber         = %d\n", frameNumber);
        printf("timeCpuCycles       = %d\n", timeCpuCycles);
        printf("numDetObj           = %d\n", *numDetObj);
        printf("numTlv              = %d\n", *numTlv);
        printf("subFrameNumber      = %d\n", *subFrameNumber);
    }
}

int parser_one_mmw_demo_output_packet(const uint8_t* data, int readNumBytes) {
    const int headerNumBytes = 40;
    const float PI = 3.14159265;

    int result = TC_PASS;

    int headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber;
    parser_helper(data, readNumBytes, &headerStartIndex, &totalPacketNumBytes, &numDetObj, &numTlv, &subFrameNumber);

    if (headerStartIndex == -1) {
        result = TC_FAIL;
        printf("************ Frame Fail, cannot find the magic words *****************\n");
    } else {
        int nextHeaderStartIndex = headerStartIndex + totalPacketNumBytes;

        if (headerStartIndex + totalPacketNumBytes > readNumBytes) {
            result = TC_FAIL;
            printf("********** Frame Fail, readNumBytes may not long enough ***********\n");
        } else if (nextHeaderStartIndex + 8 < readNumBytes && !checkMagicPattern(data + nextHeaderStartIndex)) {
            result = TC_FAIL;
            printf("********** Frame Fail, incomplete packet **********\n");
        } else if (numDetObj <= 0) {
            result = TC_FAIL;
            printf("************ Frame Fail, numDetObj = %d *****************\n", numDetObj);
        } else if (subFrameNumber > 3) {
            result = TC_FAIL;
            printf("************ Frame Fail, subFrameNumber = %d *****************\n", subFrameNumber);
        } else {
            int tlvStart = headerStartIndex + headerNumBytes;

            uint32_t tlvType = getUint32(data + tlvStart);
            uint32_t tlvLen = getUint32(data + tlvStart + 4);
            int offset = 8;

            printf("The 1st TLV\n");
            printf("    type %d\n", tlvType);
            printf("    len %d bytes\n", tlvLen);

            if (tlvType == 1 && tlvLen < totalPacketNumBytes) {
                for (int obj = 0; obj < numDetObj; obj++) {
                    float x = *((float*)(data + tlvStart + offset));
                    float y = *((float*)(data + tlvStart + offset + 4));
                    float z = *((float*)(data + tlvStart + offset + 8));
                    float v = *((float*)(data + tlvStart + offset + 12));

                    float compDetectedRange = sqrt((x * x) + (y * y) + (z * z));
                    float detectedAzimuth = (y == 0) ? ((x >= 0) ? 90 : -90) : atan(x / y) * 180 / PI;
                    float detectedElevAngle = (x == 0 && y == 0) ? ((z >= 0) ? 90 : -90) : atan(z / sqrt((x * x) + (y * y))) * 180 / PI;

                    printf("x = %f, y = %f, z = %f, v = %f, range = %f, azimuth = %f, elevAngle = %f\n",
                            x, y, z, v, compDetectedRange, detectedAzimuth, detectedElevAngle);

                    offset += 16;
                }
            }

            tlvStart = tlvStart + 8 + tlvLen;
            tlvType = getUint32(data + tlvStart);
            tlvLen = getUint32(data + tlvStart + 4);
            offset = 8;

            printf("The 2nd TLV\n");
            printf("    type %d\n", tlvType);
            printf("    len %d bytes\n", tlvLen);

            if (tlvType == 7) {
                for (int obj = 0; obj < numDetObj; obj++) {
                    uint16_t snr = getUint16(data + tlvStart + offset);
                    uint16_t noise = getUint16(data + tlvStart + offset + 2);

                    printf("snr = %d, noise = %d\n", snr, noise);

                    offset += 4;
                }
            }
        }
    }

    return result;
}

int main() {
    const char* hexPayload = "020104030605080700000603c002000043680a00ad0c0000dba6abf603000000050000000000000001000000300000007b0a943df1c8f23e00000000000000009b6c193e6a9a503f0000000000000000d1a1f73d4de17e3f0000000000000000070000000c000000de00fd03c90071039a004e030200000000020000d10e18112912c9110d104b0f790e710e6210f1126e14b814d8138712b411011127117f10e9100a12e61179108810ad105d10e80e730ef00d710e9a0e9e0e2d0e8a0e650eb80d650c2f0daf0d560dd90d2d0d5e0de30d800d8d0db60dba0dc10d410e3e0e7d0d2a0d950d740df90c320d750d610d650d750cec0cef0c280d330df80c620ca40dd50dec0c1e0c5c0c620c690cf20c4b0c860c430df60cf20c890cac0d000e950dfd0c8e0c1a0dd50c290dc50ccf0c8e0dab0df80cbf0cfe0cdb0c780cc70beb0b330de70df90dd10d850d540c810cf80cb70cf70bcf0b640b010c9a0c860c080cc00c950c6e0cb90b3c0ca80b3b0bbb0ac80bc60b990b420baa0bc60bb00a110b890b9b0b790b570b810bd70bac0b270bc90aa60b8e0bb40b850b9d0baf0a160a0a0a200af70a810ace09550a2e0b5e0bfa0aab0ad40a410a140a830ad70a780a690a340b370bc50adb096f0a320b060b480a3e09070a970a3e0a9009630a820ae709120a700ac70a0a0bf20a970af709f309d60a1c0b1a0b620a2b0adb09f009460a7409ff08ab09eb09cc0acb0ae609a9095f0a900a550a630a700aca09a609a509390a8c0ae5097d09b909230afb093f096f096e0947094809210a240aff09b609d308100904097609a5087f092a09cd083609f7084208f5077808cc08b809c40a5b0c550d5f0d390c53092f082d082109430aca0b6c0cd50c06000000180000002f110000361f000042270100000000000000000004000000090000001c00000000000000e0e205002b002b002c002c002b002e002e002f002d002d00";
    size_t dataSize = strlen(hexPayload) / 2;
    uint8_t data[dataSize];
    
    for (size_t i = 0; i < dataSize; ++i) {
        sscanf(hexPayload + 2*i, "%2hhx", &data[i]);
    }

    int result = parser_one_mmw_demo_output_packet(data, dataSize);

    if (result == TC_PASS) {
        printf("Parsing successful\n");
    } else {
        printf("Parsing failed\n");
    }

    return 0;
}
