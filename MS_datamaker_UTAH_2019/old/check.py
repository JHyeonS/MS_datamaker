import segyio

file = "FORGE_78-32_iDASv3-P11_UTC190427171923.sgy"

with segyio.open(file, "r", ignore_geometry=True) as f:
    
    dt = segyio.tools.dt(f)   # microseconds
    
    fs = 1e6 / dt
    
    print("sample interval (us):", dt)
    print("sampling rate (Hz):", fs)