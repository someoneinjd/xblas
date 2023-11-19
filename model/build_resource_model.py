#!/usr/bin/env python3

import json
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    with open("data.json") as fp:
        data = json.load(fp)
    params: list[list[float]] = []
    lut_targets: list[float] = []
    sram_targets: list[float] = []
    for test in data:
        params.append(test["params"])
        with open(test["report_path"] + "/resources/json/quartus.json") as fp:
            quartus_report = json.load(fp)
        lut_target = 0.0
        sram_target = 0.0
        for node in quartus_report["quartusFitResourceUsageSummary"]["nodes"]:
            if node["type"] == "aclsystem":
                lut_target = float(node["alm"].replace(",", ""))
                sram_target = float(node["ram"].replace(",", ""))
        lut_targets.append(lut_target)
        sram_targets.append(sram_target)
    X = np.array(params)
    y_lut = np.array(lut_targets)
    y_sram = np.array(sram_targets)
    reg = LinearRegression().fit(X, y_lut);
    print(f"LUT: {reg.coef_}")
    reg = LinearRegression().fit(X, y_sram);
    print(f"SRAM: {reg.coef_}")
