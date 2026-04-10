import segyio

path = "data/event_2417/RAW_20230719_014658.140.sgy"

with segyio.open(path, "r", ignore_geometry=True) as f:
    print("=== BASIC INFO ===")
    print("tracecount:", f.tracecount)
    print("samples:", len(f.samples))
    print("dt_us:", segyio.tools.dt(f))
    print("fs:", 1e6 / segyio.tools.dt(f))

    print("\n=== BINARY HEADER (key) ===")
    print("Sample interval (us):", f.bin[segyio.BinField.Interval])
    print("Samples per trace:", f.bin[segyio.BinField.Samples])

    print("\n=== TRACE HEADER (first trace) ===")
    th = f.header[0]
    keys = [
        segyio.TraceField.TRACE_SEQUENCE_LINE,
        segyio.TraceField.FieldRecord,
        segyio.TraceField.TraceNumber,
        segyio.TraceField.SourceX,
        segyio.TraceField.SourceY,
        segyio.TraceField.GroupX,
        segyio.TraceField.GroupY,
    ]
    for k in keys:
        print(k, th[k])