import SequenceFunctions as SF
import pandas as pd
import pathlib as pt
import time

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

###############################################
### Loop over the number of runs to perform ###
###############################################

NumberOfRuns = params.shape[0]  # Total number of runs that we want to do is number of rows in the param file

while MySequence.scanIndex < NumberOfRuns:

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

#%%
import matplotlib.pyplot as plt

plt.plot(MySequence.myCounts[f"Run{MySequence.scanIndex-1}"]["repump"])
plt.xlabel("Counter array index")
plt.ylabel("Counter array value")
