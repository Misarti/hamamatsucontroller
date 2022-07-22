
from spectroscopy import Spectrometry
from rdm_usb import RdmUsbModule
import time
import os
import json
import numpy as np
from pathlib import Path


class Spectrometry:
    def __init__(self):
        path = Path(__file__).parent.absolute()
        with open(path/  'gamma_table.json') as json_file:
            self.gamma_table = json.load(json_file)

        with open(path / "xray_coefficient_table.json") as json_file:
            self.xray_coefficient_table = json.load(json_file)

        with open(path /"xray_table.json") as json_file:
            self.xray_table = json.load(json_file)


        self.n_channels = 4095


if __name__ == "__main__":

    rdm = RdmUsbModule()
    for i in range(1000):
        rdm.get_dac()
        time.sleep(0.1)
    rdm.save_measurements()

