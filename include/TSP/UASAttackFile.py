"""
 * This file is part of UAHThesis.
 * Copyright (C) 2024 UAH - Thomas Tarter
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

from partyfirst.targetmodel.include.TSP.TSP import TSP
from partythird.constants.include.Constants import Constants
from partythird.coordtrans.include.coordtrans import Coordtrans


class UASAttackFile(TSP):
    def __init__(self, file_with_path):
        obj_name = os.path.basename(file_with_path)
        super().__init__(obj_name)

        print(f"Object Name: {obj_name}")

        data = np.genfromtxt(file_with_path, delimiter=",", skip_header=1, filling_values=0)

        # Seconds Since Liftoff(sslo)
        self.absolute_time_sslo_ = data[:, 0] * 1E-6
        lla_ddm = data[:, 1:4]

        self.pos_ecf_3m_ = Coordtrans.LLA2ECF(Constants(), lla_ddm)

        reference_point = data[0, 1:4]
        pos_enu_mmm = Coordtrans.ECF2ENU(Constants(), reference_point, self.pos_ecf_3m_)
        vel_ned_3mps = data[:, 6:9]
        vel_enu_3mps = np.vstack((vel_ned_3mps[:, 1], vel_ned_3mps[:, 0], -vel_ned_3mps[:, 2])).transpose()

        [_, self.vel_ecf_3mps_] = Coordtrans.ENU2ECF(Constants(), reference_point, pos_enu_mmm,  vel_enu_3mps)
        self.acc_ecf_3mpss_ = np.vstack(((np.diff(self.vel_ecf_3mps_, axis=0).T / np.diff(self.absolute_time_sslo_, axis=0)).T, np.array([0,0,0])))

        # ToDo: Verify this information. Plot the position, velocity and ypr and verify that they are all similar
        yaw_deg = np.rad2deg(data[:, 9])
        self.rpy_ecf_extrinsic_3deg_ = np.zeros(self.pos_ecf_3m_.shape); self.rpy_ecf_extrinsic_3deg_[:,2] = -yaw_deg


    @staticmethod
    def LoadTargetData():
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join('.','data','UAVAttackData','Simulated - OTU Survey')

        for ii in range(10):
            if not os.path.exists(os.path.join(cur_dir, abs_path)):
                cur_dir = os.path.join(cur_dir, "..")
            else:
                break

        if not os.path.exists(cur_dir):
            raise ValueError(f"Cannot find path {abs_path}")

        files = glob(os.path.join(cur_dir, '**', 'Normal','**','*vehicle_global_position_0*.csv'), recursive=True)
        tsp = []
        for ii in range(0, len(files)):
            filename = files[ii]
            tsp.append(UASAttackFile(filename))
        return tsp

if __name__ == "__main__":
    tsp_list = UASAttackFile.LoadTargetData()
    for tsp in tsp_list:
        tsp.PlotTSP()
    plt.show()