import numpy as np

def StartStopHistogram(startStop_list, timeTags, tDelay, tWindow, binNumber):
    """
    Compute a start-stop histogram for time tags within a specified time window. The window can be delayed from the start time tag by tDelay.

    Parameters:
        startStop_list (list of list): List of channel pairs to make a start stop histogram. First one is start, second is stop (e.g., [["probe", "read"]]).
        timeTags (dict): Time tags dict. Must have the structure timeTags[{channel}]=list
        tStart (list): The delay after a start tag, in seconds. Each element corresponds to one start stop hist of startStop_list.
        tWindow (list): The time duration of the histogram window, in seconds. Each element corresponds to one start stop hist of startStop_list.
        binNumber (list of int): Number of bins to divide the window into. Each element corresponds to one start stop hist of startStop_list.

    Returns:
        bin_centers (array of array): The centers of the histogram bins. Each element is for one histogram.
        histogram (array of array): The count of stop tags in each bin. Each element is for one histogram.
    """
    # Initialize function output
    X = []
    Y = []

    for idx, startStopChannels in enumerate(startStop_list):

        # Define bin edges
        binEdges = np.linspace(tDelay[idx], tDelay[idx] + tWindow[idx], binNumber[idx] + 1)
        histogram = np.zeros(binNumber[idx])

        # Create the start-window ranges
        wStart = timeTags[startStopChannels[0]] + tDelay[idx]
        wEnd = wStart + tWindow[idx]

        # Vectorize the filtering of stop tags for each window
        for w_start, w_end in zip(wStart, wEnd):
            relevantStops = timeTags[startStopChannels[1]][(timeTags[startStopChannels[1]] >= w_start) & (timeTags[startStopChannels[1]] < w_end)]
            hist, _ = np.histogram(relevantStops, bins=binEdges)
            histogram += hist

        # Compute bin centers
        bin_centers = (binEdges[:-1] + binEdges[1:]) / 2
        X.append(bin_centers)
        Y.append(histogram)

    return X, Y
