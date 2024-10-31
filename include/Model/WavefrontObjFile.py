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
import re

import numpy as np
import matplotlib.pyplot as plt


class WaveFrontObj:
  def __init__(self):
    self.num_vertices_ = None
    self.num_faces_ = None
    self.face_verts_idx_ = None
    self.vertices_ = None
    self.vert_normals_ = None
    self.face_norm_idx_ = None
    self.vert_texture_ = None
    self.face_text_idx_ = None
  pass

class WavefrontObjFile:
  def __init__(self, file_name_with_path):
    self.obj_file_ = file_name_with_path
    self.obj_name_ = os.path.basename(self.obj_file_)

    f = open(self.obj_file_, "r")
    data = f.read()
    f.close()

    [self.source_, self.mtl_file_, data] = re.split(r'mtllib ([\w.]+\n)', data)
    self.source_ = self.source_.replace('\n', '')
    self.mtl_file_ = self.mtl_file_.replace('\n', '')

    tmp = re.split(r'o (\w+)\n', data)[1:]

    self.obj_map_ = dict()
    for ii in range(0, len(tmp), 2):
      self.obj_map_[tmp[ii]] = WaveFrontObj()

      tmp_data = tmp[ii + 1]
      [[self.obj_map_[tmp[ii]].num_vertices_, self.obj_map_[tmp[ii]].num_faces_]] = re.findall('#(\d+) vertices, (\d+) faces\n', tmp_data)
      self.obj_map_[tmp[ii]].num_vertices_ = int(self.obj_map_[tmp[ii]].num_vertices_)
      self.obj_map_[tmp[ii]].num_faces_ = int(self.obj_map_[tmp[ii]].num_faces_)

      any_number = '([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)'
      face_re = '(\d+)(?:/)?(\d+)?/(\d+)'
      white_space = '\s+'
      new_line = '\n'
      generic_line = f'{white_space}{any_number}{white_space}{any_number}{white_space}{any_number}{new_line}'

      vert_pattern = f'v{generic_line}'
      self.obj_map_[tmp[ii]].vertices_ = np.array(re.findall(vert_pattern, tmp_data), dtype=np.float64)
      # self.obj_map_[tmp[ii]].vertices_ =  self.obj_map_[tmp[ii]].vertices_[:, [1,0,2]]


      norm_pattern = f'vn{generic_line}'
      self.obj_map_[tmp[ii]].vert_normals_ = np.array(re.findall(norm_pattern, tmp_data), dtype=np.float64)

      if self.obj_map_[tmp[ii]].vert_normals_.shape[0] == 0:
        self.obj_map_[tmp[ii]].vert_normals_ = None

      texture_pattern = f'vt{generic_line}'
      self.obj_map_[tmp[ii]].vert_texture_ = np.array((re.findall(texture_pattern, tmp_data)), dtype=np.float64)

      if self.obj_map_[tmp[ii]].vert_texture_.shape[0] == 0:
        self.obj_map_[tmp[ii]].vert_texture_ = None

      face_edges_pattern = f'f{white_space}{face_re}{white_space}{face_re}{white_space}{face_re}{new_line}'
      face_tmp = np.array(re.findall(face_edges_pattern, tmp_data))

      self.obj_map_[tmp[ii]].face_verts_idx_ = np.array(face_tmp[:, 0::3], dtype=np.float64)
      self.obj_map_[tmp[ii]].face_norm_idx_ = np.array(face_tmp[:, 2::3], dtype=np.float64)

      face_tmp[:, 1::3][face_tmp[:, 1::3] == ''] = -1
      self.obj_map_[tmp[ii]].face_text_idx_ = np.array(face_tmp[:, 1::3], dtype=np.float64)

  def PlotVertices(self):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for ii in self.obj_map_:
      ax.scatter(self.obj_map_[ii].vertices_[:, 0], self.obj_map_[ii].vertices_[:, 1],
                 self.obj_map_[ii].vertices_[:, 2])

    ax.set_title(f"{self.obj_name_}")
    ax.set_xlabel('X[meters]')
    ax.set_ylabel('Y[meters]')
    ax.set_zlabel('Z[meters]')
    
