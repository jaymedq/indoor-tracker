#include "AreaScannerFrame.hpp"
#include "DemoFrame.hpp"
#include "parser.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    std::vector<uint8_t> data;
    std::string inputLine;
    std::string inputFilePath = argv[1];

    // Open the input file
    std::ifstream inputFile(inputFilePath);
    if (!inputFile) {
        std::cerr << "Failed to open input file: " << inputFilePath << "\n";
        return 1;
    }

    // Read the entire content of the file into inputLine
    std::getline(inputFile, inputLine);
    inputFile.close();

    std::cout << "Input: \n" << inputLine << std::endl;

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

    auto frame = createFrame(data);
    if (frame && frame->parse(data)) {
        frame->display();
    } else {
        std::cerr << "Failed to parse frame\n";
    }

    return 0;
}
