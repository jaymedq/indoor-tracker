#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <ctime>
#include <iomanip> 
#include <chrono>
#include <unistd.h>
#include "../third-party/include/serial/serial.h"

// Define constants similar to the Python script
const std::string CONFIG_FILE_NAME = "../cfg/area_scanner_68xx_ISK.cfg";
const int REFRESH_TIME = 1000; // in milliseconds

// Class to handle IWR6843 Application logic
class IWRAPP {
public:
    IWRAPP(const std::string& cliPortName, const std::string& dataPortName)
        : cliPort(cliPortName, 115200, serial::Timeout::simpleTimeout(1000)),
          dataPort(dataPortName, 921600, serial::Timeout::simpleTimeout(1000)) {}

    void configureSensor(const std::string& configFileName) {
        std::ifstream configFile(configFileName);
        std::string line;
        while (std::getline(configFile, line)) {
            cliPort.write(line + "\n");
            std::cout << "Sent to CLI port: " << line << std::endl;
            usleep(10);
        }
    }

    std::vector<uint8_t> readData() {
        std::vector<uint8_t> byteBuffer;
        if (dataPort.available()) {
            std::string readBuffer = dataPort.read(dataPort.available());
            byteBuffer.insert(byteBuffer.end(), readBuffer.begin(), readBuffer.end());
        }
        return byteBuffer;
    }

    void run() {
        while (true) {
            std::vector<uint8_t> byteVec = readData();
            if (!byteVec.empty()) {
                std::cout << "Data received, size: " << byteVec.size() << std::endl;
                // Add parsing logic here similar to your Python parser
            }
            usleep(REFRESH_TIME);
        }
    }

private:
    serial::Serial cliPort;
    serial::Serial dataPort;
};

int main() {
    try {
        // Configure the serial ports
        IWRAPP app("COM8", "COM7"); // Use appropriate port names for Raspberry Pi and ESP32
        app.configureSensor(CONFIG_FILE_NAME);

        // Create an instance of the application and run it
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
