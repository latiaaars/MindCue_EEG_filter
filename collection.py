import time
import os
import argparse
import enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams

class BoardIds(enum.IntEnum):
    """Enum to store all supported Board Ids"""
    NO_BOARD = -100
    PLAYBACK_FILE_BOARD = -3  #:
    STREAMING_BOARD = -2  #:
    SYNTHETIC_BOARD = -1  #:
    CYTON_BOARD = 0  #:
    GANGLION_BOARD = 1  #:
    CYTON_DAISY_BOARD = 2  #:
    GALEA_BOARD = 3  #:
    GANGLION_WIFI_BOARD = 4  #:
    CYTON_WIFI_BOARD = 5  #:
    CYTON_DAISY_WIFI_BOARD = 6  #:
    BRAINBIT_BOARD = 7  #:
    UNICORN_BOARD = 8  #:
    CALLIBRI_EEG_BOARD = 9  #:
    CALLIBRI_EMG_BOARD = 10  #:
    CALLIBRI_ECG_BOARD = 11  #:
    NOTION_1_BOARD = 13  #:
    NOTION_2_BOARD = 14  #:
    GFORCE_PRO_BOARD = 16  #:
    FREEEEG32_BOARD = 17  #:
    BRAINBIT_BLED_BOARD = 18  #:
    GFORCE_DUAL_BOARD = 19  #:
    GALEA_SERIAL_BOARD = 20  #:
    MUSE_S_BLED_BOARD = 21  #:
    MUSE_2_BLED_BOARD = 22  #:
    CROWN_BOARD = 23  #:
    ANT_NEURO_EE_410_BOARD = 24  #:
    ANT_NEURO_EE_411_BOARD = 25  #:
    ANT_NEURO_EE_430_BOARD = 26  #:
    ANT_NEURO_EE_211_BOARD = 27  #:
    ANT_NEURO_EE_212_BOARD = 28  #:
    ANT_NEURO_EE_213_BOARD = 29  #:
    ANT_NEURO_EE_214_BOARD = 30  #:
    ANT_NEURO_EE_215_BOARD = 31  #:
    ANT_NEURO_EE_221_BOARD = 32  #:
    ANT_NEURO_EE_222_BOARD = 33  #:
    ANT_NEURO_EE_223_BOARD = 34  #:
    ANT_NEURO_EE_224_BOARD = 35  #:
    ANT_NEURO_EE_225_BOARD = 36  #:
    ENOPHONE_BOARD = 37  #:
    MUSE_2_BOARD = 38  #:
    MUSE_S_BOARD = 39  #:
    BRAINALIVE_BOARD = 40  #:
    MUSE_2016_BOARD = 41  #:
    MUSE_2016_BLED_BOARD = 42  #:
    EXPLORE_4_CHAN_BOARD = 44  #:
    EXPLORE_8_CHAN_BOARD = 45  #:
    GANGLION_NATIVE_BOARD = 46  #:
    EMOTIBIT_BOARD = 47  #:
    GALEA_BOARD_V4 = 48  #:
    GALEA_SERIAL_BOARD_V4 = 49  #:
    NTL_WIFI_BOARD = 50  #:
    ANT_NEURO_EE_511_BOARD = 51  #:
    FREEEEG128_BOARD = 52  #:
    AAVAA_V3_BOARD = 53 #:

'''
Collecting raw EEG with Ultracortex headset
Ubuntu serial port --> '/dev/ttyUSB0' (https://docs.openbci.com/GettingStarted/Boards/CytonGS/)
Cyton Board ID--> 0 (8 channels)
'''
parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, 
        help='serial port', required=False, default='/dev/tty.Bluetooth-Incoming-Port')
parser.add_argument('--board-id', type=int, 
        help='board id, check docs to get a list of supported boards', required=False, default=1)
args = parser.parse_args()

'''
Brainflow API: BrainFlowInputParams (parameter config)
Brainflow API: BoardShim (Access data stream from board)
'''
print('[*] Adding board settings to input config ...')
params = BrainFlowInputParams()
params.serial_port = args.serial_port
board = BoardShim(args.board_id, params)
#board = BoardShim(BoardIds.GANGLION_BOARD, params)
'''
Data collection: time --> 3 seconds (line 41 blocking call)
'''
BoardShim.enable_dev_board_logger()
print('[*] Prepairing the session ...')
board.prepare_session()
print('[*] Starting Data Stream ...')
board.start_stream()
time.sleep(3)
print('[*] Loading data ...')
data = board.get_board_data(500)
print('[*] Stop stream and release resources ...')
board.stop_stream()
board.release_session()

'''
Data extraction: 24 channels in total--EEG (8 channels)
'''
eeg_channels = BoardShim.get_eeg_channels(args.board_id)
eeg_names = BoardShim.get_eeg_names(args.board_id)

df = pd.DataFrame(np.transpose(data[:,1:]))
df_eeg = df[eeg_channels]
df_eeg.columns = eeg_names
df_eeg.plot(subplots=True, sharex=True, legend=True)
plt.legend(loc='lower right')
plt.show()

'''
Storing CSV with DatafRame and headers
'''
timestr = time.strftime("%Y%m%d-%H%M%S")
filename = timestr + '.csv'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'UltraCortex/data', filename)
df_eeg.to_csv(data_dir, sep=',', index = False)