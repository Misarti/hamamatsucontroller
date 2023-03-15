
from rdm_usb import RdmUsbModule
import time



if __name__ == "__main__":

    rdm = RdmUsbModule()
    timeout = time.time() + 300 * 1
    while time.time() <= timeout:
        #print(timeout - time.time())
        rdm.get_dac()
        time.sleep(0.1)
    isotope = 'Unknown_source'

    source = 'close_strong'
    number = '11'
    rdm.save_measurements(isotope, source, number)

