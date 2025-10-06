// parser.hpp
#ifndef PARSER_HPP
#define PARSER_HPP

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "AreaScannerFrame.hpp"
#include "DemoFrame.hpp"
#include "PeopleTrackingFrame.hpp"


extern "C" {
    __declspec(dllexport) std::unique_ptr<IFrame> createFrame(const std::vector<uint8_t> &data);
    __declspec(dllexport) int parse_frame(const uint8_t* data, size_t length);
}

#endif // PARSER_HPP
