### DEFINE AND RUN A SEQUENCE ###
import SequenceFunctions as SF
import pandas as pd
import pathlib as pt

# Path where the data will be saved
RunPath = pt.Path(r"Y:\Experiments\Rydberg\HDAWG\Initial_tests\Felix") / "Params.csv"
params = (
    pd.read_csv(RunPath)
    if RunPath.exists()
    else print("Please insert a parameter file as a csv in the folder.")
)

channel_structure = [
    "trap",
    "repump",
    "probe",
    "blue 1",
    "blue 2",
    "raman pi",
    "raman sigma",
    "dipole trap",
]

# Device information
device_id = "dev9019"
server_host = "localhost"

# Initialize sequence
MySequence = SF.RydbergSequence(params, channel_structure)

# Connect to instrument
MySequence.ConnectAndConfigure(device_id, server_host)

# Add blocks
MySequence.LoadMOT()
MySequence.doMolasses()
MySequence.WaitDuration(5e-6)
MySequence.doMemory()
MySequence.doRepetition(
    round(
        MySequence.params["Interrogation time [s]"].loc[MySequence.scanIndex]
        / MySequence.subSequenceDuration
    )
)


# Upload C code to sequencer
MySequence.UploadSequencer()
MySequence.RunSequence()

#print(MySequence.cCode)