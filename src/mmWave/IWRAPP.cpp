#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <ctime>
#include <iomanip>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <chrono>
#include <algorithm>
#include "serial/serial.h"
#include "parser.hpp"
#include "IWRApp.hpp"

// Class to handle IWR6843 Application logic
    IWRAPP::IWRAPP(const std::string &cliPortName, const std::string &dataPortName)
        : cliPort(cliPortName, 115200, serial::Timeout::simpleTimeout(1000)),
          dataPort(dataPortName, 921600, serial::Timeout::simpleTimeout(1000)) {}

    void IWRAPP::configureSensor(const std::string &configFileName)
    {
        std::ifstream configFile(configFileName);
        std::string line;
        while (std::getline(configFile, line))
        {
            cliPort.write(line + "\n");
            std::cout << "Sent to CLI port: " << line << std::endl;
#ifdef _WIN32
            Sleep(10); // Sleep for 10 milliseconds
#else
            usleep(10000); // Sleep for 10 milliseconds
#endif
        }
    }

    std::vector<uint8_t> IWRAPP::readData()
    {
        std::vector<uint8_t> byteBuffer(0,0);
        if (dataPort.available())
        {
            std::string readBuffer = dataPort.read(dataPort.available());
            byteBuffer.insert(byteBuffer.end(), readBuffer.begin(), readBuffer.end());
        }
        return byteBuffer;
    }

    void IWRAPP::run(const std::string &filename)
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
                    frame->toCsv(filename);
                }
            }
#ifdef _WIN32
            Sleep(REFRESH_TIME * 1000); // Sleep for REFRESH_TIME seconds
#else
            sleep(REFRESH_TIME); // Sleep for REFRESH_TIME seconds
#endif
        }
    }
