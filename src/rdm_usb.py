import ctypes
import os
import coloredlogs, logging
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
import bisect
import matplotlib
import matplotlib.pyplot as plt
import csv
import time

@dataclass_json
@dataclass
class Eeprom:
    comp_level: int
    energy_lower: int
    energy_upper: int
    convert_usv: int
    adc_to_kev: int

@dataclass_json
@dataclass
class RdmUsb:
    success: int
    invalid_handle: int
    unsuccess: int
    invalid_value: int
    not_updated: int

@dataclass_json
@dataclass
class EnergyParams:
    lower_energy: ctypes.c_uint16
    area_start: ctypes.c_uint16
    area_end: ctypes.c_uint16
    base_sievert: ctypes.c_uint16
    base_energy: ctypes.c_uint16



class RdmUsbModule:
    def __init__(self):
        logging.basicConfig()
        self.logger = logging.getLogger('RdmUsb')
        coloredlogs.install(level='INFO', logger=self.logger)
        #self.invalid_handle_value = -1
        self.eeprom = Eeprom(comp_level=10,
                             energy_lower=12,
                             energy_upper=14,
                             convert_usv=16,
                             adc_to_kev=18)
        self.rdmusb = RdmUsb(success=0,
                             invalid_handle=1,
                             unsuccess=2,
                             invalid_value=3,
                             not_updated=4)
        self.lib = self.get_functions()
        self.device_handle = self.open_device()
        self.energy, self.base_sievert, self.base_energy = self.eeprom_info()
        self.count_rate = 0
        self.histsave = np.empty((65535))
        self.histgraph = np.empty((4096))
        self.log_listX, self.log_corr = self.get_log_corr()
        self.area_lower = bisect.bisect_left(self.log_listX, self.energy.area_start.value)
        self.area_upper = bisect.bisect_right(self.log_listX, self.energy.area_end.value)
        self.sievert = 0
        self.sievert_array = []
        self.event_count = 0


    def get_functions(self):
        file_path = os.path.abspath("./res/rdmusb.dll")
        lib = ctypes.WinDLL(file_path)
        return lib

    def open_device(self):
        device_handle = self.lib.RdmUsb_OpenDevice()
        if device_handle == -1:
            self.logger.error('Device not found')
        else:
            self.logger.info('Device connected')
        return device_handle

    def close_device(self):
        self.lib.RdmUsb_CloseDevice(self.device_handle)
        self.logger.info('Device disconnected')

    def get_info_from_eeprom(self, parameter, data):
        read = self.lib.RdmUsb_ReadEeprom
        result = read(self.device_handle, parameter, ctypes.byref(data))

    def eeprom_info(self):
        lower_energy = ctypes.c_uint16()
        area_start = ctypes.c_uint16()
        area_end = ctypes.c_uint16()
        base_siev = ctypes.c_uint16()
        base_ener = ctypes.c_uint16()
        energy_params = [lower_energy, area_start, area_end, base_siev, base_ener]
        eeprom_params = self.eeprom.to_dict()

        for e, c in zip(eeprom_params.values(), energy_params):
            self.get_info_from_eeprom(e, c)
        energy = EnergyParams(*energy_params)

        d_base_siev = ctypes.c_double(base_siev.value/1000)
        d_base_ener = ctypes.c_double(base_ener.value/1000)

        return energy, d_base_siev, d_base_ener

    def energy_threshold(self):
        energy_threshold = ctypes.c_ushort(15)
        self.lib.RdmUsb_GetEnergyThreshold(self.device_handle, ctypes.byref(energy_threshold))
        return energy_threshold

    def read_major(self):
        """

        :return:
        """
        major = ctypes.c_uint16()
        self.get_info_from_eeprom(self.eeprom.ver_major, major)
        print('Major', major)

    def get_log_corr(self):
        file_path =  "./res/gf_C12137_2.csv"
        log_listX = []
        log_corr = []
        with open(file_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                log_listX.append(float(row[0])*self.base_energy.value)
                log_corr.append(float(row[1]))
        return log_listX, log_corr

    def sievert_per_sec(self, sievert):
        sievert_per_sec = sievert * 0.0036
        self.sievert_array.append(sievert_per_sec)
        mean = sum(self.sievert_array)/len(self.sievert_array)
        result = mean * self.base_sievert.value
        print(result)

    def get_dac(self):
        index = ctypes.c_ushort()
        buffer_inside = ctypes.c_ushort()
        size = ctypes.c_ushort()
        seq = ctypes.c_ushort*1048
        buffer = [buffer_inside]*1048
        arr = seq(*buffer)
        temperature = ctypes.c_double()
        result = self.lib.RdmUsb_GetDacDataAndTemperature(self.device_handle, ctypes.byref(size), arr,
                                                          ctypes.byref(index), ctypes.byref(temperature))
        self.event_count +=1
        if size != 0:
            for i in range(size.value):
                self.count_rate += size.value
                addr = arr[i]
                self.histsave[addr] += 1
                binary_shift_addr = addr >> 4
                self.histgraph[binary_shift_addr] += 1
                if self.area_lower < binary_shift_addr < self.area_upper:
                    self.sievert += self.log_corr[binary_shift_addr]
        if self.event_count % 10 == 0:
            self.sievert_per_sec(self.sievert)
            self.sievert = 0











