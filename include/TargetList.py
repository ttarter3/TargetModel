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

from partyfirst.targetmodel.include.Target import Target


class TargetList():
  def __init__(self):
    self.targets_ = []

  def AddTarget(self, target: Target):
    assert(isinstance(target, Target) or isinstance(target, TargetList))

    if isinstance(target, TargetList):
      self.targets_.extend(target.targets_)
    else: self.targets_.append(target)