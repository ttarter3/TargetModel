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

import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from config.Config import Config
from partyfirst.GUID import GUID
from partyfirst.targetmodel.include.Model.WavefrontObjFile import WavefrontObjFile
from partyfirst.targetmodel.include.RCS.RCS import RCS
from partyfirst.targetmodel.include.RCS.RCSProfile import RCSProfile
from partysecond.supportfunctions.SupportFunctions.FilesAndDirectories.FilesAndDirectories import FindFiles
from partythird.constants.include.Constants import Constants

# conda did not work when trying to install tinyobjloader
# only thing that worked was downloading microsoft build tools -> https://visualstudio.microsoft.com/visual-cpp-build-tools/ -> Download build tools -> run exe -> next -> next -> next ...
# Then running ".\pip install tinyobjloader==2.0.0rc7" which contains precompiled windows libraries
# newer versions of tinyobjloader did not work. i.e. 2.0.0.rc9, idk about rc8
# import tinyobjloader
# tinyobjloader and pyassimp did not work.  just parse it ourselves.

obj_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
for ii in range(10):
  obj_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.sep.join(['..'] * ii), 'data')

  if os.path.exists(obj_file_path):
    break

obj_files = FindFiles(obj_file_path, '*.obj')

class Model3D(RCS):
  def __init__(self, object_name: str, object_wavefront_obj_file: str) -> object:
      """
      Generic Model for hosting a 3d Object
      :param object_name: UniqueId for the object of interest. Multiple objects of similar builds would have 'airplane1', 'airplane2'
      """
      self.object_name_ = object_name
      self.guid_ = GUID.generate_guid()

      self.wavefront_obj_ = WavefrontObjFile(object_wavefront_obj_file)

      tmp = np.vstack([self.wavefront_obj_.obj_map_[key].vertices_ for key in self.wavefront_obj_.obj_map_])
      self.reference_vertice_ = np.median(tmp, axis=0)
      self.reference_vertice_[0] = np.max(tmp[:,0])

  @staticmethod
  def GetModel3D(object_name: str) -> object:
    assert(isinstance(object_name, str))

    for file_name in obj_files:
      if object_name in os.path.basename(file_name):
        return Model3D(object_name, file_name)

    logging.error("Returning default model(DJIPantomSmoothed) for object({}) because {}.obj model was not found in {}".format(object_name, object_name, obj_file_path))
    default_obj_file = obj_files[0]
    default_obj = os.path.basename(default_obj_file)
    return Model3D(default_obj, default_obj_file)

  def GetRCS(self, roll_deg, yaw_deg):
    # ToDo: use some logic here to convert all the indices to position

    # roll_deg = np.ones_like(roll_deg) * 45
    # yaw_deg = np.ones_like(yaw_deg) * 0

    if np.all(np.isnan(roll_deg)) or np.all(np.isnan(yaw_deg)):
      return RCSProfile(0.0
                        , np.array((0, 1))
                        , -1 * np.inf * np.ones(2))

    assert(not np.any(np.isnan(roll_deg)) or not np.any(np.isnan(yaw_deg)))

    r = R.from_euler('xz', np.vstack((roll_deg, yaw_deg)).T, degrees=True).as_matrix()
    r_rot = np.einsum('ijk->ijk', r)


    if Config.config("plot")["tgt_profile_hist"]:
      rng_prof_fig = plt.figure()
    if Config.config("plot")["tgt_pre_and_post_rotation"]:
      d3_plot = plt.figure()
      d3_plot_ax1 = d3_plot.add_subplot(121, projection='3d')
      d3_plot_ax2 = d3_plot.add_subplot(122, projection='3d')

    x_data = np.array(())

    reference_vert = self.reference_vertice_
    reference_trans_m = r_rot @ reference_vert

    for obj_name in self.wavefront_obj_.obj_map_:
      # ToDo: Remove the ::25 in the line below and replace with :, this is because we only have 8GB of ram on the test machine
      verts = self.wavefront_obj_.obj_map_[obj_name].vertices_[::25, :]

      v_rot     = np.einsum('ij->ji', verts)
      obj_trans = np.einsum('ijk->ikj', r_rot @ v_rot)

      hist, bin_edges = np.histogram(obj_trans[:, :, 0], 100, density=True)

      if Config.config("plot")["tgt_profile_hist"]:
        plt.figure(rng_prof_fig)
        plt.plot(bin_edges[:-1], hist, label=obj_name)
        plt.legend()

      if Config.config("plot")["tgt_pre_and_post_rotation"]:
        plt.figure(d3_plot)
        plt.title("Target Pre and Post Rotation")
        idx = 0
        d3_plot_ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2], label=obj_name)
        d3_plot_ax2.scatter(obj_trans[idx, :, 0], obj_trans[idx, :, 1], obj_trans[idx, :, 2], label=obj_name)
        # Set the labels
        d3_plot_ax1.set_xlabel('X')
        d3_plot_ax1.set_ylabel('Y')
        d3_plot_ax1.set_zlabel('Z')
        d3_plot_ax2.set_xlabel('X')
        d3_plot_ax2.set_ylabel('Y')
        d3_plot_ax2.set_zlabel('Z')
        # Set the title
        d3_plot_ax1.set_title('Original')
        d3_plot_ax1.legend()
        d3_plot_ax2.set_title(f'Rotated R({roll_deg[idx]:.2f})Y({yaw_deg[idx]:.2f})')

      x_data = np.append(x_data, obj_trans[:, :, 0])


    hist, bin_edges = np.histogram(x_data, 100, density=True)
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    if Config.config("plot")["all_obj_point_hist_as_profile"]:
      rng_prof_fig2 = plt.figure()
      plt.vlines(np.median(reference_trans_m[:,0]), min(hist), max(hist), 'r', label="Reference Pt")
      plt.plot(center, hist, label="obj histogram")
      plt.title("All Object Point Histogram as Profile")
      plt.xlabel("down range(m)")
      plt.ylabel("Count")
      plt.legend()


    one_profile_ref_time = np.median(reference_trans_m[:, 0], axis=0) / Constants.C('mps')
    one_profile_tgt_time = center / Constants.C('mps')
    one_profile_tgt_signal = hist
    threshold = 10 * np.finfo(np.float32).eps
    one_profile_tgt_signal[one_profile_tgt_signal < threshold] = threshold

    rcs_p = RCSProfile(one_profile_ref_time, one_profile_tgt_time, one_profile_tgt_signal)

    return rcs_p


if __name__ == "__main__":
  model_3d = Model3D.GetModel3D("DJIPantomSmoothed")
  model_3d.wavefront_obj_.PlotVertices()
  plt.show()

