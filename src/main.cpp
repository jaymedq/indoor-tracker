#include "AreaScannerFrame.hpp"
#include "DemoFrame.hpp"
#include "parser.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "IWRAPP.hpp"
#include <algorithm>
#include <iomanip>
#include <chrono>

std::string getfileName()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *local_time = std::localtime(&now_time);

    std::ostringstream timeStream;
    timeStream << std::put_time(local_time, "%d-%m-%Y %H:%M:%S");
    std::string fileName = "output_" + timeStream.str() + ".csv";
    // Replace spaces and colons with underscores for file name compatibility
    std::replace(fileName.begin(), fileName.end(), ' ', '_');
    std::replace(fileName.begin(), fileName.end(), ':', '_');
    return fileName;
}

int main(int argc, char *argv[])
{
    std::string cliPortName = "COM4";
    std::string dataPortName = "COM3";
    std::string filename = "";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc)
        {
            filename = argv[++i];
        }
        else if (cliPortName.empty())
        {
            cliPortName = arg;
        }
        else if (dataPortName.empty())
        {
            dataPortName = arg;
        }
        else
        {
            std::cerr << "Unexpected argument: " << arg << std::endl;
            return 1;
        }
    }

    if (cliPortName.empty() || dataPortName.empty())
    {
        std::cerr << "Usage: " << argv[0] << " <CLI_COM_PORT> <DATA_COM_PORT> [-o output.csv]" << std::endl;
        return 1;
    }

    if (filename.empty())
    {
        filename = getfileName();
    }

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return 1;
    }

    std::cout << "Writing to file: " << filename << std::endl;
    file << "numObj,x,y,z,velocity,numObj_static,x_static,y_static,z_static,velocity_static,timestamp\n";
    file.close();

    try
    {
        IWRAPP app(cliPortName, dataPortName);
        app.configureSensor(app.CONFIG_FILE_NAME);
        app.run(filename);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
