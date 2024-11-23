import zhinst.toolkit as tk
import numpy as np
import zhinst
from pathlib import Path
from datetime import datetime
import time
import sys


class SequenceClass:
    def __init__(self, parameter_file, channel_structure, experimentPath):
        # Load parameter file
        self.params = parameter_file
        self.experimentPath = experimentPath

        # Initialize C code string
        self.cCode = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}
        self.lastAdded_cCode = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}

        # Initialize C code wave definition strings
        self.defWave = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}
        self.defWave = {
            key: value + "wave square32 = ones(32);\nwave marker32 = marker(32,1);\n" for key, value in self.defWave.items()
        }  # Add a sqaure32 and marker32 definition to all AWG

        # Initialize indices
        self.scanIndex = int(0)  # Indicate what row of the parameter file we are at
        self.waveIndices = {
            "gauss": 0,
            "square": 0,
            "risingGauss": 0,
            "fallingGauss": 0,
        }  # Indicate how many times we have defined a wave with these properties
        self.loopIndex = int(0)  # subscript to add when defining a while loop index in C

        # Initialize isConnected
        self.isConnected = False

        # Upload channel structure (what AOM is connected to what channel entry)
        self.channel_structure = channel_structure

        # Define which built-in waveform to use. for homemade waveforms.
        self.waveBuiltIn = {
            "gauss": "gauss",
            "square": "ones",
            "risingGauss": "gauss",
            "fallingGauss": "gauss",
        }
        self.myTimetags = {f"Run{i}": {f"{channel}": [] for channel in channel_structure} for i in range(len(self.params))}

        self.myCounts = {f"Run{i}": {f"{channel}": [] for channel in channel_structure} for i in range(len(self.params))}

    def ConnectAndConfigure(self, device_id, server_host):
        """
        Establishes a connection to a specified instrument via a data server.
        Then updates some preset.

        This method connects to the data server using the provided host address and
        then connects to the specified device identified by its unique device ID. It
        sets up channel grouping, output range, and initializes the settings for
        both AWG cores and output channels.

        Args:
            device_id (str): The unique identifier of the device to connect to.
            server_host (str): The hostname or IP address of the data server.

        Returns:
            None
        """
        self.session = tk.Session(server_host)  ## connect to data server
        self.device = self.session.connect_device(device_id)  ## connect to device
        self.device_id = device_id
        self.server_host = server_host

        self.grouping = 0  # Channel grouping 2x4
        self.output_range = 1.0  # Output range [V]

        # Configure the HDAWG
        with self.device.set_transaction():
            self.device.oscs[0].freq(1e6)
            self.device.oscs[1].freq(1e6)
            self.device.oscs[2].freq(1e6)
            self.device.oscs[3].freq(1e6)

            self.device.awgs[0].synchronization.enable(1)
            self.device.awgs[1].synchronization.enable(1)
            self.device.awgs[2].synchronization.enable(1)
            self.device.awgs[3].synchronization.enable(1)
            self.device.system.synchronization.source("internal")

        for awgGroup in np.arange(4):

            awg_cores_i = awgGroup * 2**self.grouping + np.arange(2**self.grouping)  # AWG cores
            channels_i = awgGroup * 2 ** (self.grouping + 1) + np.arange(2 ** (self.grouping + 1))  # Output channels

            awg_cores = [self.device.awgs[awg_core] for awg_core in awg_cores_i]
            channels = [self.device.sigouts[channel] for channel in channels_i]

            # Per-core settings
            with self.device.set_transaction():

                # Grouping mode
                self.device.system.awg.channelgrouping(self.grouping)

                for awg_core in awg_cores:
                    awg_core.outputs[0].gains[0](1.0)  # Set the output gains matrix to identity
                    awg_core.outputs[0].gains[1](0.0)
                    awg_core.outputs[1].gains[0](0.0)
                    awg_core.outputs[1].gains[1](1.0)
                    awg_core.outputs[0].modulation.mode(1)  # Turn off modulation mode
                    awg_core.outputs[1].modulation.mode(2)

                # Per-channel settings
                for channel in channels:
                    channel.range(self.output_range)  # Select the output range
                    channel.on(True)  # Turn on the outputs. Should be the last setting

        self.daq = zhinst.core.ziDAQServer(server_host, 8004, 6)
        self.isConnected = True

    def InitCounter(self, channel_list):
        for counter, channel in enumerate(channel_list):
            triggerInput = int(self.channel_structure.index(channel) + 32)
            # Configure the pulse counter
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/mode", 4)  # Timetagging mode
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/inputselect", triggerInput)  # Get timetag from Channel = triggerInput - 31
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/trigrising", 1)  # Get event at rising edge
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/trigfalling", 0)  # Don't get event at falling edge
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/integrate", 1)  # Sum up values over time
            self.daq.setDouble(f"/{self.device_id}/cnts/{counter}/period", 1e-7)  # Minimum time between each event !
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/enable", 0)  # Disable measurement
            self.daq.subscribe(f"/{self.device_id}/cnts/{counter}/sample")
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/enable", 1)

            self.isCounterInitialized = True
            self.clockbase = float(self.daq.getInt(f"/{self.device_id}/clockbase"))

    def GatherData(self, channel_list, pollTime):
        timeTags = {}
        countNumber = {}
        counterNumber = np.arange(len(channel_list))

        # Start the poll of the timetags
        counterData = self.daq.poll(pollTime, 500, 0, True)

        # Extract the data and store it
        for counter in counterNumber:
            channel = self.channel_structure[int(self.daq.getInt(f"/{self.device_id}/cnts/{counter}/inputselect") - 32)]
            try:
                counterData_i = counterData[f"/{self.device_id}/cnts/{counter}/sample"]
                timeTags[f"{channel}"] = counterData_i["timestamp"] / self.clockbase
                countNumber[f"{channel}"] = counterData_i["counter"]
            except:
                print("\nWARNING : counter " + str(counter) + " empty")
                timeTags[f"{channel}"] = []
                countNumber[f"{channel}"] = []

            self.myTimetags[f"Run{self.scanIndex}"][f"{channel}"] = np.concatenate((self.myTimetags[f"Run{self.scanIndex}"][f"{channel}"], timeTags[f"{channel}"]))
            self.myCounts[f"Run{self.scanIndex}"][f"{channel}"] = np.concatenate((self.myCounts[f"Run{self.scanIndex}"][f"{channel}"], countNumber[f"{channel}"]))

    def StartRecordingData(self, channel_list, pollTime):

        # Create the folder to store data if it doesn't exist
        current_time = datetime.now().strftime("%H_%M_%S")
        dataPath = self.experimentPath / current_time
        dataPath.mkdir(parents=True, exist_ok=True)

        # Gather the data
        while self.device.awgs[0].sequencer.status():  # While Sequencer is running
            self.GatherData(channel_list, pollTime)

        self.GatherData(channel_list, 5)  # Gather data one last time in case we missed some

    def StopRecordingData(self, channel_list):
        counterNumber = np.arange(len(channel_list))
        for counter in counterNumber:
            self.daq.unsubscribe(f"/{self.device_id}/cnts/{counter}/sample")
            self.daq.setInt(f"/{self.device_id}/cnts/{counter}/enable", 0)  # Switch off counting

    def UploadSequencer(self):
        """
        Uploads the current sequence to the Arbitrary Waveform Generator (AWG).

        It will combine the C code of the wave definitions written in self.defWave and the instruction in self.cCode

        This method checks if there is an active connection to the data server before attempting to upload
        the sequence. If the connection is established, it sets the appropriate parameters on the AWG module
        and compiles the sequence from the defined waveforms.

        Raises:
            ConnectionError: If there is no connection to the data server.
                             Ensure that self.ConnectToInstrument is called first.

        Returns:
            None
        """
        if self.isConnected:
            self.device.awgs[0].load_sequencer_program(str(self.defWave["Seq1"]) + str(self.cCode["Seq1"]))
            self.device.awgs[1].load_sequencer_program(str(self.defWave["Seq2"]) + str(self.cCode["Seq2"]))
            self.device.awgs[2].load_sequencer_program(str(self.defWave["Seq3"]) + str(self.cCode["Seq3"]))
            self.device.awgs[3].load_sequencer_program(str(self.defWave["Seq4"]) + str(self.cCode["Seq4"]))
        else:
            raise ConnectionError("Cannot load AWG because there is no connection to the data server. Use self.ConnectToInstrument() first!")

    def RunSequence(self):
        """
        Run the sequence. You need to upload the sequencers first !

        TO ADD : A wait command until the run is over
        """
        with self.device.set_transaction():
            self.device.awgs[0].enable(True, deep=True)
            self.device.awgs[1].enable(True, deep=True)
            self.device.awgs[2].enable(True, deep=True)
            self.device.awgs[3].enable(True, deep=True)

        print(f"Running #{self.scanIndex}.")

    def ResetSequence(self):
        """
        Erase all previous sequence instructions, for all sequencers.
        """
        # Initialize C code strings
        self.cCode = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}

        # Initialize wave definition strings
        self.defWave = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}

        # Add a square32 and marker32 definition to all AWG
        self.defWave = {key: value + "wave square32 = ones(32);\nwave marker32 = marker(32,1);\n" for key, value in self.defWave.items()}

    def OpenAOM(self, channel_list, duration, frequencies, amplitudes, setFreq=True):
        """
        Opens AOM lines for extended durations within a sequence.

        Args:
            channel_list (list of str): Specifies which channels to open.
            duration (float): Duration of the pulse in seconds.
            frequencies (list of float): Frequencies in MHz, must match the length of channel_list.
            amplitudes (list of float): Amplitudes in volts, must match the length of channel_list.

        Returns:
            None. Updates the self.cCode attribute with the necessary commands.
        """
        # Initialize local cCode.
        cCode = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}

        # Initialize the counter of frequency changes
        freqChangeCounter = [0] * 4

        # Calculate hold time and add playHold commands
        samples = TimeToSample(duration)
        hold_blocks = samples // 240000000
        remaining_samples = samples % 240000000
        remaining_samples = remaining_samples - 32 if remaining_samples > 32 else remaining_samples

        # Create a boolean array for active channels
        channels = [ch in channel_list for ch in self.channel_structure]

        # Set frequencies and amplitudes with 0 for inactive channels
        freq = RearrangeList(frequencies, channel_list, self.channel_structure)
        amps = RearrangeList(amplitudes, channel_list, self.channel_structure)

        # Build command string for setting frequencies. Need to sandwich the command by 2 times 96 samples, for it to work well.
        for ch, active in enumerate(channels):
            if active and setFreq:
                cCode[f"Seq{ChannelToSequencer(ch)}"] += "playHold(96);\n" + f'setInt("sines/{ch}/harmonic", {int(freq[ch])});\n' + "playHold(96);\n"
                freqChangeCounter[ChannelToSequencer(ch) - 1] += 1

        # Don't forget to add the 96 samples on other sequencers, even if their frequency hasn't changed. Otherwise, synchronization problems.
        if setFreq:
            for seq, totalChanges in enumerate(freqChangeCounter):
                numOfAdjustmentNeeded = int(max(freqChangeCounter))
                cCode[f"Seq{seq+1}"] += "playHold(96);\n" * 2 * (numOfAdjustmentNeeded - totalChanges)

        # Build the waveform playback command
        wave = [f"{int(active)}*(square32 + marker32)*{float(amps[idx])}" for idx, active in enumerate(channels)]

        # Finally, add all the waveforms in the corresponding channels. zeros will be sent if channel is not selected.
        for seq in [1, 2, 3, 4]:
            channelIndex = 2 * (seq - 1)
            cCode[f"Seq{seq}"] += f"playWave({wave[channelIndex]},{wave[channelIndex+1]});\n"
            cCode[f"Seq{seq}"] += "playHold(240000000);\n" * hold_blocks
            cCode[f"Seq{seq}"] += f"playHold({remaining_samples});\nwaitWave();\n"

            # Update the global cCode for each sequencer seq
            self.cCode[f"Seq{seq}"] += cCode[f"Seq{seq}"]
            self.lastAdded_cCode[f"Seq{seq}"] += cCode[f"Seq{seq}"]

    def AddWaveform(self, channel_list, offsetDuration, waveDuration, waitDuration, waveType, channelFreq, waveParam, waveAmp, numPerChannel, setFreq=True):
        """
        Adds waveforms to the sequencer for specified channels.

        This method constructs waveform C-code definitions based on the provided parameters,
        organizes them according to the specified channel order, and generates corresponding
        C code to update on the sequencer. It ensures that all parameters have
        compatible lengths and transforms input parameters into nested lists.

        Note :
            Markers are sent along with the waveform. Perhaps this needs to be changed !

        Args:
            channel_list (list of str): A list of channels to which the waveforms will be added.
                                        Must correspond to the entries in self.channel_structure.
            offsetDuration (list):      A list or nested list specifying the duration to offset each waveform.
            waveDuration (list):        A list or nested list specifying the duration of each waveform.
            waitDuration (list):        A list or nested list specifying the wait time after each waveform.
            waveType (list):            A list or nested list specifying the type of each waveform
                                        (e.g., "gauss", "square", "risingGauss", "fallingGauss").
            channelFreq (list):            A list specifying the frequency of each channel.
            waveParam (list):           A list or nested list specifying additional parameters for the waveform types.
            waveAmp (list):             A list or nested list specifying the amplitude of each waveform.
            numPerChannel (list):       A list indicating the number of waveforms to be added per channel.

        Raises:
            ValueError: If the lengths of the parameters do not match the number of waveforms specified in numPerChannel.

        Returns:
            None: Updates self.lastAdded_cCode and self.cCode with the generated C code for waveform playback.

        Example:
            # Adding waveforms for the channels "blue 1" and "trap" with specified durations and types
            instance.AddWaveform(
                channel_list=['blue 1', 'trap'],
                offsetDuration=[1e-6, [0.2e-6, 1e-6]],
                waveDuration=[0.3e-6, [0.4e-6, 5e-7]],
                waitDuration=[0.1e-6, [0.1e-6, e-6]],
                waveType=['gauss', ['square', 'gauss']],
                channelFreq=[100, 250],
                waveParam=[2e-6, [1e-6, 3e-6]],
                waveAmp=[1, [0.5e-6, 1e-6]],
                numPerChannel=[1, 2]
            )
        """
        # Initialize local C code
        cCode = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}

        # Create a boolean array for active channels
        channels = [ch in channel_list for ch in self.channel_structure]

        # Check that all parameters have the right length
        if (
            not sum(numPerChannel)
            == CountElements(offsetDuration)
            == CountElements(waveDuration)
            == CountElements(waitDuration)
            == CountElements(waveType)
            == CountElements(waveParam)
            == CountElements(waveAmp)
        ):
            raise ValueError("Waveforms not added because lengths of arguments do not match.")

        # Ensure that parameters are nested list
        (offsetDuration, waveDuration, waitDuration, waveType, channelFreq, waveParam, waveAmp,) = toNestedList(
            offsetDuration,
            waveDuration,
            waitDuration,
            waveType,
            channelFreq,
            waveParam,
            waveAmp,
        )

        # Initialize the playWave argument, where all 8 channels are sending zeroes
        wave = [f"zeros({TimeToSample(sum(offsetDuration[0])+sum(waveDuration[0])+sum(waitDuration[0]))})"] * 8
        totalSampleNumber = [TimeToSample(sum(offsetDuration[0]) + sum(waveDuration[0]) + sum(waitDuration[0]))] * 8

        # Frequency change counter
        freqChangeCounter = [0] * 4

        # Rearrange all the lists in order for each channel
        (numPerChannel, offsetDuration, waveDuration, waitDuration, waveType, channelFreq, waveParam, waveAmp,) = [
            RearrangeList(param, channel_list, self.channel_structure)
            for param in [
                numPerChannel,
                offsetDuration,
                waveDuration,
                waitDuration,
                waveType,
                channelFreq,
                waveParam,
                waveAmp,
            ]
        ]

        # Loop over each channel
        for idx, active in enumerate(channels):
            if active:
                waveToJoin = []
                markerToJoin = []
                waveIndex = 0
                totalSampleNumber[idx] = 0
                # Loop over the number of waveforms for the channel number idx+1
                for j in np.arange(numPerChannel[idx]):

                    # Loop variables
                    sampleNumber = TimeToSample(waveDuration[idx][waveIndex])
                    sampleWidth = TimeToSample(waveParam[idx][waveIndex])
                    currentWaveType = waveType[idx][waveIndex]
                    offsetSampleNumber = TimeToSample(offsetDuration[idx][waveIndex])
                    waitSampleNumber = TimeToSample(waitDuration[idx][waveIndex])

                    # Update totalSampleNumber, to track the total number of samples added with each waveform, including the offset, wait time, and the zeroes.
                    totalSampleNumber[idx] += sampleNumber + offsetSampleNumber + waitSampleNumber

                    # Define each wave and marker! Bigass f-string is unreadable. Not sure how to implement it differently, probably need to write a function
                    self.defWave[f"Seq{ChannelToSequencer(idx)}"] += (
                        f"wave {currentWaveType}{self.waveIndices[currentWaveType]}"  # Wave type and index (to avoid defining functions with same name). Example: wave gauss0 = ...
                        + f" = {self.waveBuiltIn[currentWaveType]}("  # wave gauss0 = gauss( ...
                        + f'{sampleNumber}{", 1" if self.waveBuiltIn[currentWaveType] == "gauss" else ""}'  # wave gauss0 = gauss(sampleNumber, 1, ...
                        + f'{f", {sampleNumber/2}" if currentWaveType == "gauss" else (f", {sampleNumber}" if currentWaveType == "risingGauss" else (f", 0" if currentWaveType == "fallingGauss" else ""))}'
                        + f'{f", {sampleWidth}" if self.waveBuiltIn[currentWaveType] == "gauss" else ""});\n'
                    )

                    # Add it to the defWave cCode
                    self.defWave[
                        f"Seq{ChannelToSequencer(idx)}"
                    ] += f"wave marker_{currentWaveType}{self.waveIndices[currentWaveType]} = marker({TimeToSample(waveDuration[idx][waveIndex])}, 1);\n"

                    # Define one wave of the corresponding channel (add the offset before, and the waiting time after)
                    waveToJoin.append(
                        f"{waveAmp[idx][waveIndex]}"
                        + f'*{f"join(zeros({offsetSampleNumber}), " if offsetDuration!=0 else ("join(" if waitDuration!=0 else "")}'
                        + f"{currentWaveType}"
                        + f"{self.waveIndices[currentWaveType]}"
                        + f'{f", zeros({waitSampleNumber}))" if waitDuration!=0 else (f")" if offsetDuration!=0 else "")}'
                    )

                    # Same for the marker
                    markerToJoin.append(
                        f'{f"join(zeros({offsetSampleNumber}), " if offsetDuration!=0 else ("join(" if waitDuration!=0 else "")}'
                        + f"marker_{currentWaveType}{self.waveIndices[currentWaveType]}"
                        + f'{f", zeros({waitSampleNumber}))" if waitDuration!=0 else (f")" if offsetDuration!=0 else "")}'
                    )

                    self.waveIndices[currentWaveType] += 1
                    waveIndex += 1

                # Join waveform and markers to obtain the final waveform/marker for each channel
                if len(waveToJoin) > 1:
                    wave[idx] = f"join({', '.join(waveToJoin)}) + join({', '.join(markerToJoin)})"
                else:
                    wave[idx] = f"{waveToJoin[0]} + {markerToJoin[0]}"

                # Update the frequency, and (IMPORTANT) add the buffer time needed by the node to be adjusted
                if setFreq:
                    if channelFreq[idx] != False:
                        cCode[f"Seq{ChannelToSequencer(idx)}"] += "playZero(96);\n" + f'setInt("sines/{idx}/harmonic", {int(channelFreq[idx][0])});\n' + "playZero(96);\n"
                        freqChangeCounter[ChannelToSequencer(idx) - 1] += 1

        if setFreq:
            for seq, totalChanges in enumerate(freqChangeCounter):
                numOfAdjustmentNeeded = int(max(freqChangeCounter))
                cCode[f"Seq{seq+1}"] += "playZero(96);\n" * 2 * (numOfAdjustmentNeeded - totalChanges)

        if len(set([x for x in totalSampleNumber if x != 0])) > 1:
            print("Warning ! Discrepency in number of samples between channels. Might result in non-synchronized sequence. Try to put times in multiples of 20ns.")
            print(f"Number of samples outputted on each channel : {totalSampleNumber}")

        # Finally, play all the waveforms in the corresponding channels. zeros will be sent if channel is not selected.
        for seq in [1, 2, 3, 4]:
            channelIndex = 2 * (seq - 1)
            cCode[f"Seq{seq}"] += f"playWave({wave[channelIndex]},{wave[channelIndex+1]});\n"

            # Update the global cCode
            self.cCode[f"Seq{int(seq)}"] += cCode[f"Seq{int(seq)}"]
            self.lastAdded_cCode[f"Seq{int(seq)}"] += cCode[f"Seq{int(seq)}"]

    def RampAOM(self, channel_list, duration, startFreq, stopFreq, startAmp, stopAmp, rampType, rampParam, rampSteps=100, setFreq=True):

        # Initialize local cCode.
        cCode = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}

        # Initialize the counter of frequency changes
        freqChangeCounter = [0] * 4

        # Initiate C loop
        for seq in [1, 2, 3, 4]:
            cCode[f"Seq{seq}"] += f"cvar i{self.loopIndex};\n" + f"for (i{self.loopIndex} = 0; i{self.loopIndex} < {int(rampSteps)}; i{self.loopIndex}++) " + "{\n"

        # Create a boolean array for active channels
        channels = [ch in channel_list for ch in self.channel_structure]

        # Set frequencies, amplitudes and other params with 0 for inactive channels
        startFreq = RearrangeList(startFreq, channel_list, self.channel_structure)
        stopFreq = RearrangeList(stopFreq, channel_list, self.channel_structure)
        startAmp = RearrangeList(startAmp, channel_list, self.channel_structure)
        stopAmp = RearrangeList(stopAmp, channel_list, self.channel_structure)
        rampType = RearrangeList(rampType, channel_list, self.channel_structure)
        rampParam = RearrangeList(rampParam, channel_list, self.channel_structure)

        # Initiate parameters
        wave = ["zeros(32)"] * 8
        # Frequency change counter
        freqChangeCounter = [0] * 4

        # FOR loop over each channel
        for ch, active in enumerate(channels):
            if active:
                if rampType[ch] == "Exp":
                    # A*exp(-t/tau) + B
                    freqA = int(startFreq[ch] - stopFreq[ch])
                    freqB = int(stopFreq[ch])
                    Tau = (rampParam[ch] / duration) * 100
                    ampA = startAmp[ch] - stopAmp[ch]
                    ampB = stopAmp[ch]

                    wave[ch] = f"({ampA} * exp(-1*i{self.loopIndex}/{Tau}) + {ampB}) * square32 + marker32"

                # Update frequency change counter (because one frequency change command needs to be sandwich by some waiting time)
                if setFreq:
                    if startFreq[ch] != stopFreq[ch]:
                        freqChangeCounter[ChannelToSequencer(ch) - 1] += 1
                        cCode[f"Seq{ChannelToSequencer(ch)}"] += (
                            "\tplayHold(96);\n" + f'\tsetInt("sines/{ch}/harmonic", ({freqA} * exp(-1*i{self.loopIndex}/{Tau}) + {freqB}));\n' + "\tplayHold(96);\n"
                        )

        numOfAdjustmentNeeded = int(max(freqChangeCounter))
        stepSamples = round(TimeToSample(duration) / rampSteps / 16) * 16
        for seq, totalChanges in enumerate(freqChangeCounter):
            channelIndex = 2 * (seq)
            cCode[f"Seq{seq+1}"] += "\tplayHold(96);\n" * 2 * (numOfAdjustmentNeeded - totalChanges)
            cCode[f"Seq{seq+1}"] += f"\tplayWave({wave[channelIndex]},{wave[channelIndex+1]});\n"
            cCode[f"Seq{seq+1}"] += f"\tplayHold({stepSamples - numOfAdjustmentNeeded * 2 * 96});\n"
            cCode[f"Seq{seq+1}"] += "}\n"

            # Update the global cCode
            self.cCode[f"Seq{seq+1}"] += cCode[f"Seq{seq+1}"]
            self.lastAdded_cCode[f"Seq{seq+1}"] += cCode[f"Seq{seq+1}"]

        self.loopIndex += 1

    def WaitDuration(self, waitTime):
        waitPeriods = TimeToSequencerPeriod(waitTime)
        cCode = f"wait({waitPeriods});\n"
        for seq in [1, 2, 3, 4]:
            self.cCode[f"Seq{int(seq)}"] += cCode
            self.lastAdded_cCode[f"Seq{int(seq)}"] += cCode


class DLCZSequence(SequenceClass):
    def LoadMOT(self):
        self.OpenAOM(
            ["Trap", "Repump", "Bfield"],
            self.params["Loading time [s]"].loc[self.scanIndex],
            [
                self.params["Trap Freq [MHz]"].loc[self.scanIndex],
                self.params["Repump Freq [MHz]"].loc[self.scanIndex],
            ],
            [
                self.params["Trap Power [V]"].loc[self.scanIndex],
                self.params["Repump Power [V]"].loc[self.scanIndex],
            ],
        )

    def Molasses(self):
        # First ramp phase
        self.RampAOM(
            ["Trap", "Repump"],
            self.params["Molasse time [s]"].loc[self.scanIndex],
            [
                self.params["Trap Freq [MHz]"].loc[self.scanIndex],
                self.params["Repump Freq [MHz]"].loc[self.scanIndex],
            ],
            [
                self.params["Molasse final Trap Freq [MHz]"].loc[self.scanIndex],
                self.params["Molasse final Repump Freq [MHz]"].loc[self.scanIndex],
            ],
            [
                self.params["Trap Power [V]"].loc[self.scanIndex],
                self.params["Repump Power [V]"].loc[self.scanIndex],
            ],
            [
                self.params["Molasse final Trap Power [V]"].loc[self.scanIndex],
                self.params["Molasse final Repump Power [V]"].loc[self.scanIndex],
            ],
            ["Exp"],
            [self.params["Molasse tau [s]"].loc[self.scanIndex], self.params["Molasse tau [s]"].loc[self.scanIndex]],
        )

        # Hold phase
        self.OpenAOM(
            ["Trap", "Repump"],
            self.params["Molasse hold time [s]"].loc[self.scanIndex],
            [
                self.params["Molasse final Trap Freq [MHz]"].loc[self.scanIndex],
                self.params["Molasse final Repump Freq [MHz]"].loc[self.scanIndex],
            ],
            [
                self.params["Molasse final Trap Power [V]"].loc[self.scanIndex],
                self.params["Molasse final Repump Power [V]"].loc[self.scanIndex],
            ],
        )

    def ZeemanPumping(self):
        # Zeeman + Read + Repump ON
        self.OpenAOM(
            ["Repump", "ZeemanPump", "Read"],
            self.params["Zeeman Pumping time [s]"].loc[self.scanIndex],
            [
                self.params["Repump Freq [MHz]"].loc[self.scanIndex],
                self.params["Zeeman pump Freq [MHz]"].loc[self.scanIndex],
                self.params["Read Zeeman Freq [MHz]"].loc[self.scanIndex],
            ],
            [
                self.params["Repump Power [V]"].loc[self.scanIndex],
                self.params["Zeeman pump Power [V]"].loc[self.scanIndex],
                self.params["Read Zeeman Power [V]"].loc[self.scanIndex],
            ],
        )

        # Read + Repump ON (but Zeeman pump OFF)
        self.OpenAOM(
            ["Repump", "Read"],
            self.params["Zeeman Pumping buffer time [s]"].loc[self.scanIndex],
            [
                self.params["Repump Freq [MHz]"].loc[self.scanIndex],
                self.params["Read Zeeman Freq [MHz]"].loc[self.scanIndex],
            ],
            [
                self.params["Repump Power [V]"].loc[self.scanIndex],
                self.params["Read Zeeman Power [V]"].loc[self.scanIndex],
            ],
        )

        # Wait a small amount
        self.WaitDuration(1e-6)

    def doDLCZ(self):
        self.subSequenceDuration = 100e-9 + self.params["Storage time [s]"].loc[self.scanIndex] + 400e-9 + 100e-9 + self.params["Clean duration [s]"].loc[self.scanIndex]

        # Write pulse
        self.AddWaveform(
            ["Write"],
            [0],
            [100e-9],
            [0],
            ["gauss"],
            [self.params["Write Freq [MHz]"].loc[self.scanIndex]],
            [self.params["Write FHWM [s]"].loc[self.scanIndex] / 2.355],  # Adjusted for FHWM to sigma
            [self.params["Write Power [V]"].loc[self.scanIndex]],
            [1],
        )

        # Storage time
        self.WaitDuration(self.params["Storage time [s]"].loc[self.scanIndex])

        # Read pulse
        self.AddWaveform(
            ["Read"],
            [0],
            [400e-9],
            [0],
            ["gauss"],
            [self.params["Read Freq [MHz]"].loc[self.scanIndex]],
            [self.params["Read FHWM [s]"].loc[self.scanIndex] / 2.355],  # Adjusted for FHWM to sigma
            [self.params["Read Power [V]"].loc[self.scanIndex]],
            [1],
        )

        # Small wait time of 100 ns
        self.WaitDuration(100e-9)

        # Clean pulse
        self.AddWaveform(
            ["ZeemanPump", "Read"],
            [0, 0],
            [self.params["Clean duration [s]"].loc[self.scanIndex], self.params["Clean duration [s]"].loc[self.scanIndex]],
            [0, 0],
            ["square", "square"],
            [self.params["Zeeman Clean Freq [MHz]"].loc[self.scanIndex], self.params["Read Clean Freq [MHz]"].loc[self.scanIndex]],
            [0, 0],  # waveParam argument has no effect for square waveforms
            [self.params["Zeeman Clean Power [V]"].loc[self.scanIndex], self.params["Read Clean Power [V]"].loc[self.scanIndex]],
            [1, 1],
        )

    def StartOfSubSequenceRepetition(self):
        # Update lastAdded_cCode if repetition is needed
        self.lastAdded_cCode = {"Seq1": str(), "Seq2": str(), "Seq3": str(), "Seq4": str()}

    def EndOfSubSequenceRepetition(self, numberRepetitions):
        for seq in np.arange(4) + 1:
            if self.cCode[f"Seq{int(seq)}"].endswith(self.lastAdded_cCode[f"Seq{int(seq)}"]):

                # Remove the last added piece of code
                self.cCode[f"Seq{int(seq)}"] = self.cCode[f"Seq{int(seq)}"][: -len(self.lastAdded_cCode[f"Seq{int(seq)}"])]

                # Update C Code with while loop
                self.cCode[f"Seq{int(seq)}"] += (
                    f"var repetitionIndex{self.loopIndex} = 0;\n"
                    + f"while (repetitionIndex{int(self.loopIndex)} < {int(numberRepetitions)}) "
                    + "{\n\t"
                    + self.lastAdded_cCode[f"Seq{int(seq)}"].replace("\n", "\n\t")
                    + f"repetitionIndex{self.loopIndex}++;\n"
                    + "}\n"
                )
        self.loopIndex += 1

    def TotalRepetitions(self, numberRepetitions):
        for seq in np.arange(4) + 1:
            # Update C Code with while loop
            self.cCode[f"Seq{int(seq)}"] = (
                f"var repetitionIndex{self.loopIndex} = 0;\n"
                + f"while (repetitionIndex{int(self.loopIndex)} < {int(numberRepetitions)}) "
                + "{\n\t"
                + self.cCode[f"Seq{int(seq)}"].replace("\n", "\n\t")
                + f"repetitionIndex{self.loopIndex}++;\n"
                + "}\n"
            )
        self.loopIndex += 1


def TimeToSample(time_duration):
    return int(round(time_duration * 2.4e9 / 16) * 16)


def TimeToSequencerPeriod(time_duration):
    return int(round(time_duration * 2.4e9 / 16) * 16 / 8)


def CountElements(inputList):
    """Counts the total number of elements in a list, whether it's nested or not."""
    return sum(len(sublist) if isinstance(sublist, list) else 1 for sublist in inputList)


def toNestedList(*args):
    """Transforms multiple lists into nested lists, ensuring every element in each list is a list."""
    return [[elem if isinstance(elem, list) else [elem] for elem in lst] for lst in args]


def RearrangeList(lst, channel_list, self_channel_structure):
    """
    Rearranges a list based on the mapping of another list to align with a specified structure.

    Given an array of values and a list of channels, this function rearranges the
    values so that they align with the specified channel structure. If a channel
    in the structure does not have a corresponding value, it will insert False in
    its place.

    Parameters:
        lst (list): A list of values where each value corresponds to a channel in
                     the channel_list. This can include nested lists.
        channel_list (list): A list of channel names corresponding to the values in lst.
                             This list defines the order of the elements in lst.
        self_channel_structure (list): The desired channel structure that dictates
                                       the output order of the values.

    Returns:
        list: A new list where the values from arr are rearranged to match the order
              of self_channel_structure. If a channel is not present in lst, False
              is placed in the corresponding position.

    Example:
        channel_list = ['blue 1', 'trap']
        lst = [[2, 7], 3]
        self_channel_structure = ['trap', 'repump', 'probe', 'blue 1', 'blue 2']

        result = rearrange_arr(lst, channel_list, self_channel_structure)
        # result will be: [3, False, False, [2, 7], False]
    """
    # Create a mapping of channel_list with arr using zip
    channel_map = dict(zip(channel_list, lst))
    # Rearrange arr based on the order of self.channel_structure
    rearranged_arr = [channel_map.get(ch, False) for ch in self_channel_structure]
    return rearranged_arr


def ChannelToSequencer(channel):
    return int(channel // 2 + 1)
