#include "serial/serial.h"

class IWRAPP
{
public:
    IWRAPP(const std::string &cliPortName, const std::string &dataPortName);

    void configureSensor(const std::string &configFileName);

    std::vector<uint8_t> readData();

    void run(const std::string &filename);

private:
    serial::Serial cliPort;
    serial::Serial dataPort;

public:
    // Define constants similar to the Python script
    const std::string CONFIG_FILE_NAME = "cfg/area_scanner_68xx_ISK.cfg";
    const int REFRESH_TIME = 1; // in seconds
};
