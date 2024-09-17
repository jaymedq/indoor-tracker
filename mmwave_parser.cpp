#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <src/frames/AreaScannerFrame.hpp>
#include <src/frames/DemoFrame.hpp>

// Factory function to create appropriate frame
std::unique_ptr<Frame> createFrame(const std::vector<uint8_t> &data) {
  if (data.size() < 8) {
    std::cerr << "Invalid data size\n";
    return nullptr;
  }
//   if (data[0] == 0x02 && data[1] == 0x01 && data[2] == 0x04 &&
//       data[3] == 0x03) {
//     return std::make_unique<DemoFrame>();
//   } else 
  if (data[0] == 0x02 && data[1] == 0x01 && data[2] == 0x04 &&
             data[3] == 0x03) { // Update based on actual magic word check for
                                // AreaScannerFrame
    return std::make_unique<AreaScannerFrame>();
  } else {
    std::cerr << "Unrecognized frame type\n";
    return nullptr;
  }
}

int main() {
  std::vector<uint8_t> data;
  std::string inputLine;

  // Open the input file
  std::ifstream inputFile("input.txt");
  if (!inputFile) {
    std::cerr << "Failed to open input file.\n";
    return 1;
  }

  // Read the entire content of the file into inputLine
  std::getline(inputFile, inputLine);
  inputFile.close();

  std::cout << "Input: \n" << inputLine << std::endl;

  // Check if the input length is even
  if (inputLine.length() % 2 != 0) {
    std::cerr << "Invalid input length. Hex stream should have an even number "
                 "of characters.\n";
    return 1;
  }

  // Process the input string in pairs of two characters
  for (size_t i = 0; i < inputLine.length(); i += 2) {
    std::string hexByte = inputLine.substr(i, 2);

    // Convert hex string to uint8_t
    uint8_t byte = static_cast<uint8_t>(std::stoul(hexByte, nullptr, 16));
    data.push_back(byte);
  }

  auto frame = createFrame(data);
  if (frame && frame->parse(data)) {
    frame->display();
  } else {
    std::cerr << "Failed to parse frame\n";
  }

  return 0;
}
