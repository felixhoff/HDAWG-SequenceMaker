import SequenceFunctions as SF
import pandas as pd
import pathlib as pt
import time
import numpy as np

# Path where the data will be saved
experimentPath = pt.Path(r"Y:\Experiments\Rydberg\HDAWG\Initial_tests\Felix\Test")
if not (experimentPath / "Params.csv").exists():
    raise FileNotFoundError("Please insert a parameter file as a CSV in the folder.")
params = pd.read_csv(experimentPath / "Params.csv")

# Channel structure of the HDAWG. Should be a list of strings saying which beam is controled by what channel.
# For example ["trap", "repump"] means that the trapping beam is controlled by channel 1 and repump is controlled by channel 2.
channel_structure = [
    "Trap",  # CH1
    "Repump",  # CH2
    "Bfield",  # CH3
    "ZeemanPump",  # CH4
    "Write",  # CH5
    "Read",  # CH6
    "Spare 1",  # CH7
    "Spare 2",  # CH8
]

# Device information
device_id = "dev9019"
server_host = "10.3.20.143"

# Initialize sequence and connect to instrument
MySequence = SF.RydbergSequence(params, channel_structure, experimentPath)
MySequence.ConnectAndConfigure(device_id, server_host)

# Plot Params
MySequence.StartStop = [["Write", "Read"], ["ZeemanPump", "Read"]]
MySequence.timeAfterStart = [1e-6, 1e-3]
MySequence.windowDuration = [100e-6, 50e-6]
MySequence.binNumber = [100, 100]
MySequence.Y = [np.zeros(binNumber) for binNumber in MySequence.binNumber]

###############################################
### Loop over the number of runs to perform ###
###############################################

while MySequence.scanIndex < MySequence.NumberOfRuns:

    # Reset sequence instance
    MySequence.ResetSequence()

    # Add main blocks
    MySequence.LoadMot()
    MySequence.Molasses()
    MySequence.ZeemanPumping()

    # Sub sequence block
    MySequence.StartOfSubSequenceRepetition()
    MySequence.doDLCZ()
    numberOfReps = round(MySequence.params["Interrogation time [s]"].loc[MySequence.scanIndex] / MySequence.subSequenceDuration)
    MySequence.EndOfSubSequenceRepetition()

    # Total number of repetitions

    # Upload C code to sequencer
    MySequence.UploadSequencer()

    # Run and Measure
    MySequence.InitCounter(["Repump"])
    MySequence.RunSequence()
    MySequence.StartRecordingData(["Repump"], 0.1)
    MySequence.StopRecordingData(["Repump"])

    # Increment
    MySequence.scanIndex += 1
