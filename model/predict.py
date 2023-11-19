#!/usr/bin/env python3

import json
from scipy.optimize import linprog

if __name__ == "__main__":
    data = {}
    resource = {}
    with open("model.json", "r") as fp:
        data = json.load(fp)
    with open("resource.json", "r") as fp:
        resource = json.load(fp)
    c: list[float] = []
    lut: list[float] = []
    dsp: list[float] = []
    sram: list[float] = []
    all_kernels = list(data.keys())[0:-1]
    for kernel in all_kernels:
        if kernel not in data["critical_path"]:
            c += [0.0 for _ in data[kernel]["params"]]
        else:
            c += data[kernel]["time"]
        lut += data[kernel]["lut"]
        dsp += data[kernel]["dsp"]
        sram += data[kernel]["sram"]
    res = linprog(
        [-i for i in c],
        A_ub=[lut, dsp, sram],
        b_ub=[resource["lut"], resource["dsp"], resource["sram"]],
        bounds=[(0, None) for _ in lut],
    )
    i = 0
    for kernel in all_kernels:
        print(f"{kernel}: [", end="")
        for p in data[kernel]["params"]:
              print(f"{p} = {res.x[i]}, ", end="")
              i += 1
        print("]")
