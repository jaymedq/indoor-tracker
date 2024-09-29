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
#include "parser.hpp"

// Define constants similar to the Python script
const std::string CONFIG_FILE_NAME = "../cfg/area_scanner_68xx_ISK.cfg";
const int REFRESH_TIME = 1; // in seconds

// Class to handle IWR6843 Application logic
class IWRAPP
{
public:
    IWRAPP(const std::string &cliPortName, const std::string &dataPortName)
        : cliPort(cliPortName, 115200, serial::Timeout::simpleTimeout(1000)),
          dataPort(dataPortName, 921600, serial::Timeout::simpleTimeout(1000)) {}

    void configureSensor(const std::string &configFileName)
    {
        std::ifstream configFile(configFileName);
        std::string line;
        while (std::getline(configFile, line))
        {
            cliPort.write(line + "\n");
            std::cout << "Sent to CLI port: " << line << std::endl;
            usleep(10000); // Sleep for 10 milliseconds
        }
    }

    std::vector<uint8_t> readData()
    {
        std::vector<uint8_t> byteBuffer;
        if (dataPort.available())
        {
            std::string readBuffer = dataPort.read(dataPort.available());
            byteBuffer.insert(byteBuffer.end(), readBuffer.begin(), readBuffer.end());
        }
        return byteBuffer;
    }

    void run()
    {
        while (true)
        {
            std::vector<uint8_t> byteVec = readData();
            if (!byteVec.empty())
            {
                std::cout << "Data received, size: " << byteVec.size() << std::endl;
                auto frame = createFrame(byteVec);
                if (!(frame && frame->parse(byteVec)))
                {
                    std::cerr << "Failed to parse frame\n";
                }
                else
                {
                    frame->toCsv("output.csv");
                }
            }
            sleep(REFRESH_TIME); // Sleep for REFRESH_TIME microseconds
        }
    }

private:
    serial::Serial cliPort;
    serial::Serial dataPort;
};

int main(int argc, char *argv[])
{
    // Check if the correct number of arguments is provided
    std::string cliPortName = "COM8";
    std::string dataPortName = "COM7";
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <CLI_COM_PORT> <DATA_COM_PORT>" << std::endl;
    }
    else
    {
        std::string cliPortName = argv[1];
        std::string dataPortName = argv[2];
    }


    try
    {
        IWRAPP app(cliPortName, dataPortName);
        app.configureSensor(CONFIG_FILE_NAME);
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
