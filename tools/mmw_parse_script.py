import csv
from datetime import datetime
import serial
import time
import numpy as np
import os
import sys
# import the parser function 
from parser_mmw_demo import parser_one_mmw_demo_output_packet

# Time between frames in seconds
REFRESH_TIME = 1

# Change the configuration file name
configFileName = 'area_scanner_68xx_ISK.cfg'

# Change to output to csv
OUTPUT_TO_CSV = True
outpout_file_path = configFileName[:-4]+'_output.csv'
# Change the debug variable to use print()
DEBUG = True

# Constants
maxBufferSize = 2**15
CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0
maxBufferSize = 2**15
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
detObj = {}  
frameData = {}    
currentIndex = 0
# word array to convert 4 bytes to a 32 bit number
word = [1, 2**8, 2**16, 2**24]

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyUSB0', 115200)
    # Dataport = serial.Serial('/dev/ttyUSB1', 921600)
    
    # # Windows
    CLIport = serial.Serial('COM8', 115200)
    Dataport = serial.Serial('COM7', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)
        
    return CLIport, Dataport

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
                
            digOutSampleRate = int(splitWords[11])
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = int(splitWords[5])

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters

##################################################################################
# USE parser_mmw_demo SCRIPT TO PARSE ABOVE INPUT FILES
##################################################################################
def readAndParseData14xx(Dataport, configParameters):
    #load from serial
    global byteBuffer, byteBufferLength

    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
    
    # Check that the buffer has some data
    if byteBufferLength > 16:
        
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:
            
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]
                
            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word)
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    
    # If magicOK is equal to 1 then process the message
    if magicOK:
        # Read the entire buffer
        readNumBytes = byteBufferLength
        if(DEBUG):
            print("readNumBytes: ", readNumBytes)
        allBinData = byteBuffer
        if(DEBUG):
            print("allBinData: ", allBinData[0], allBinData[1], allBinData[2], allBinData[3])

        # init local variables
        totalBytesParsed = 0
        numFramesParsed = 0

        # parser_one_mmw_demo_output_packet extracts only one complete frame at a time
        # so call this in a loop till end of file
        #             
        # parser_one_mmw_demo_output_packet function already prints the
        # parsed data to stdio. So showcasing only saving the data to arrays 
        # here for further custom processing
        parser_result, \
        headerStartIndex,  \
        totalPacketNumBytes, \
        numDetObj,  \
        numTlv,  \
        subFrameNumber,  \
        detectedX_array,  \
        detectedY_array,  \
        detectedZ_array,  \
        detectedV_array,  \
        detectedRange_array,  \
        detectedAzimuth_array,  \
        detectedElevation_array,  \
        detectedSNR_array,  \
        detectedNoise_array = parser_one_mmw_demo_output_packet(allBinData[totalBytesParsed::1], readNumBytes-totalBytesParsed,DEBUG)

        # Check the parser result
        if(DEBUG):
            print ("Parser result: ", parser_result)
            print(readBuffer.hex())
        if (parser_result == 0): 
            totalBytesParsed += (headerStartIndex+totalPacketNumBytes)    
            numFramesParsed+=1
            if(DEBUG):
                print("totalBytesParsed: ", totalBytesParsed)
                print("Raw Bytes:", bytes(allBinData[:totalBytesParsed]).hex())
            ##################################################################################
            # TODO: use the arrays returned by above parser as needed. 
            # For array dimensions, see help(parser_one_mmw_demo_output_packet)
            # help(parser_one_mmw_demo_output_packet)
            ##################################################################################

            
            # For example, dump all S/W objects to a csv file
            """"""
            import csv
            if (numFramesParsed == 1):
                democsvfile = open('mmw_demo_output.csv', 'w', newline='')                
                demoOutputWriter = csv.writer(democsvfile, delimiter=',',
                                        quotechar='', quoting=csv.QUOTE_NONE)                                    
                demoOutputWriter.writerow(["frame","DetObj#","x","y","z","v","snr","noise","raw"])            
            
            for obj in range(numDetObj):
                demoOutputWriter.writerow([numFramesParsed-1, obj, detectedX_array[obj],\
                                            detectedY_array[obj],\
                                            detectedZ_array[obj],\
                                            detectedV_array[obj],\
                                            detectedSNR_array[obj],\
                                            detectedNoise_array[obj],\
                                            bytes(allBinData[:totalBytesParsed]).hex()])
            """"""
            detObj = {"numObj": numDetObj, "range": detectedRange_array, \
                        "x": detectedX_array, "y": detectedY_array, "z": detectedZ_array, "raw": bytes(allBinData[:totalBytesParsed]).hex()}
            dataOK = 1 
        else: 
            # error in parsing; exit the loop
            # print("error in parsing this frame; continue")
            pass

        
        shiftSize = totalPacketNumBytes            
        byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
        byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
        byteBufferLength = byteBufferLength - shiftSize
        
        # Check that there are no errors with the buffer length
        if byteBufferLength < 0:
            byteBufferLength = 0
        # All processing done; Exit
        if(DEBUG):
            print("numFramesParsed: ", numFramesParsed)

    return dataOK, frameNumber, detObj


class IWRAPP:

    def __init__(self, parent=None):
        pass

    # Funtion to update the data and display in the plot
    def update(self):
        
        dataOk = 0
        global detObj
        x = []
        y = []
        z = []
        
        # Read and parse the received data
        dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)
        import pprint
        if dataOk and len(detObj["x"]) > 0:
            pprint.pp(detObj)
            x = detObj["x"]
            y = detObj["y"]
            z = detObj['z']
            
            if OUTPUT_TO_CSV:
                # Check if the file exists, and create it with headers if it doesn't
                if not os.path.isfile(outpout_file_path):
                    create_csv_with_headers(outpout_file_path, list(detObj.keys()) + ['Timestamp'])
                append_dict_to_csv(outpout_file_path, detObj)

        return dataOk, x, y, z


    def onNewData(self):
        
        # Update the data and check if the data is okay        
        dataOk,newx,newy, newz = self.update()

        if dataOk:
            # Store the current frame into frameData
            #frameData[currentIndex] = detObj
            #currentIndex += 1
            x = newx
            y = newy
            z = newz
            print(f"{x=},{y=},{z=}")
        return dataOk
            
    
    def run(self):
        while(1):
            dataOk = self.onNewData()
            if(dataOk):
                time.sleep(REFRESH_TIME)
            else:
                time.sleep(0.1)



def create_csv_with_headers(file_path, headers):
    # Create the CSV file with headers if it doesn't exist
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

def append_dict_to_csv(file_path, data_dict):
    # Add a timestamp entry to the dictionary
    data_dict['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        
        # Write the data
        writer.writerow(data_dict)

def main():        
    # Configurate the serial port
    CLIport, Dataport = serialConfig(configFileName)

    # Get the configuration parameters from the configuration file
    global configParameters 
    configParameters = parseConfigFile(configFileName)

    app = IWRAPP()
    try:
        app.run()
    except Exception as e:
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        raise e

if __name__ == "__main__":
    main()

