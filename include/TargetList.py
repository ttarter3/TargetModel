'''
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
'''

import numpy as np

from config.Config import Config
from partyfirst.targetmodel.include.Model.Model3D import Model3D
from partyfirst.targetmodel.include.Target import Target


class TargetList():
  def __init__(self):
    self.targets_ = []

  def AddTarget(self, target: Target):
    assert(isinstance(target, Target) or isinstance(target, TargetList))

    if isinstance(target, TargetList):
      self.targets_.extend(target.targets_)
    else: self.targets_.append(target)

  def CalcTargetPointingAngle(self, pos_ecef_xyz_m):
    # NOTE: Point at the first target, not always the case
    med = np.median(self.targets_[0].tsp_.pos_ecf_3m_, axis=0)
    pointing_vector = med - pos_ecef_xyz_m
    point = pointing_vector / np.linalg.norm(pointing_vector)
    return point

  @staticmethod
  def Load():
    if Config.config("target_set")["type"] == "uav" and Config.config("target_set")["relative_path"] is None:
      from partyfirst.targetmodel.include.TSP.UASAttackFile import UASAttackFile

      tgt_list = TargetList()
      sub_tgt_list = UASAttackFile.LoadTargetData()
      for tsp in sub_tgt_list:
        rcs = Model3D.GetModel3D(tsp.object_name_)
        tgt = Target(tsp, rcs)
        tgt_list.AddTarget(tgt)
      return tgt_list

    else: raise ValueError("Unsupported Target Set")

