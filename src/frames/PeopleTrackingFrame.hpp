/*
 *   Copyright (c) 2025 Jayme Queiroz CPGEI
 *   All rights reserved.
 */
#if !defined(PEOPLE_TRACKING_FRAME_HPP)
#define PEOPLE_TRACKING_FRAME_HPP

#include "IFrame.hpp"

class PeopleTrackingFrame : public IFrame
{
public:
    static const uint32_t NUM_RADAR_TXANT = 3; // 2 transmitting antennas
    static const uint32_t NUM_RADAR_RXANT = 4; // 4 receiving antennas

    static const uint32_t MAX_DYNAMIC_CFAR_PNTS =          150;
    static const uint32_t MAX_STATIC_CFAR_PNTS =           150;
    static const uint32_t DOA_OUTPUT_MAXPOINTS =           (MAX_DYNAMIC_CFAR_PNTS * 4 + MAX_STATIC_CFAR_PNTS);
    static const uint32_t MAX_RESOLVED_OBJECTS_PER_FRAME = DOA_OUTPUT_MAXPOINTS;
    static const uint32_t MAX_NUM_RANGE_BINS =             (256);

    bool parse(std::vector<uint8_t> &data) override;

    void display() const override;

    void toCsv(const std::string &path) const;

    /*!
     * @brief
     *  Message header for reporting detection information from data path.
     *
     * @details
     *  The structure defines the message header.
     */
    typedef struct MmwDemo_output_message_header_t
    {
        /*! @brief   Output buffer magic word (sync word). It is initialized to  {0x0102,0x0304,0x0506,0x0708} */
        uint16_t magicWord[4];

        /*! brief   Version: : MajorNum * 2^24 + MinorNum * 2^16 + BugfixNum * 2^8 + BuildNum   */
        uint32_t version;

        /*! @brief   Total packet length including header in Bytes */
        uint32_t totalPacketLen;

        /*! @brief   platform type */
        uint32_t platform;

        /*! @brief   Frame number */
        uint32_t frameNumber;

        /*! @brief   Time in CPU cycles when the message was created. For XWR16xx/XWR18xx: DSP CPU cycles, for XWR14xx: R4F CPU cycles */
        uint32_t timeCpuCycles;

        /*! @brief   Number of detected objects */
        uint32_t numDetectedObj;

        /*! @brief   Number of TLVs */
        uint32_t numTLVs;

        /*! @brief   For Advanced Frame config, this is the sub-frame number in the range
         * 0 to (number of subframes - 1). For frame config (not advanced), this is always
         * set to 0. */
        uint32_t subFrameNumber;
    } MmwDemo_output_message_header;

    /*!
    * @brief
    * Structure holds the message body for the  Point Cloud units
    *
    * @details
    * Reporting units for range, azimuth, and doppler
    */
    typedef struct MmwDemo_output_message_compressedPoint_unit_t
    {
        /*! @brief elevation  reporting unit, in radians */
        float elevationUnit;
        /*! @brief azimuth  reporting unit, in radians */
        float azimuthUnit;
        /*! @brief Doppler  reporting unit, in m/s */
        float dopplerUnit;
        /*! @brief range reporting unit, in m */
        float rangeUnit;
        /*! @brief SNR  reporting unit, linear */
        float snrUint;

    } MmwDemo_output_message_compressedPoint_unit;

    /*!
    * @brief
    * Structure holds the message body to UART for the  Point Cloud
    *
    * @details
    * For each detected point, we report range, azimuth, and doppler
    */
    typedef struct MmwDemo_output_message_compressedPoint_t
    {
        /*! @brief Detected point elevation, in number of azimuthUnit */
        int8_t elevation;
        /*! @brief Detected point azimuth, in number of azimuthUnit */
        int8_t azimuth;
        /*! @brief Detected point doppler, in number of dopplerUnit */
        int16_t doppler;
        /*! @brief Detected point range, in number of rangeUnit */
        uint16_t range;
        /*! @brief Range detection SNR, in number of snrUnit */
        uint16_t snr;

    } MmwDemo_output_message_compressedPoint;

    // TLV header structure
    struct MmwDemo_output_message_tl_t {
        uint32_t type;
        uint32_t length;
    };

    typedef struct MmwDemo_output_message_compressedPointCloud_uart_t
    {
        MmwDemo_output_message_tl_t                 header;
        MmwDemo_output_message_compressedPoint_unit pointUint;
        MmwDemo_output_message_compressedPoint      point[MAX_RESOLVED_OBJECTS_PER_FRAME];
    } MmwDemo_output_message_compressedPointCloud_uart;

private:
    void parseTLV(std::vector<uint8_t> payload, uint32_t type, uint32_t length);
};
#endif // PEOPLE_TRACKING_FRAME_HPP