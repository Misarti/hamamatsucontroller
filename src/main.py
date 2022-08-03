
from rdm_usb import RdmUsbModule
from find_nuclide import NuclideIdentifier
import time
import os
import json
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    n = NuclideIdentifier()
    rdm = RdmUsbModule()
    for i in range(1000):
        rdm.get_dac()
        time.sleep(0.1)
    rdm.save_measurements()
    rdm.find_peaks()

