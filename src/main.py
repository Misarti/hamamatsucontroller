from rdm_usb import RdmUsbModule
import time
import schedule


if __name__ == "__main__":

    rdm = RdmUsbModule()
    rdm.eeprom_info()
    schedule.every(2).seconds.do(rdm.get_dac)
    while True:
        schedule.run_pending()
        time.sleep(1)

