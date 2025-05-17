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
import numpy as np
import simplekml
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from partyfirst.GUID import GUID
from partythird.constants.include.Constants import Constants
from partythird.coordtrans.include.coordtrans import Coordtrans, Rotations

# Seconds Since LiftOff(sslo)
class TSP(object):
    def __init__(self, object_name):
        self.object_name_ = object_name
        self.guid_ = GUID.generate_guid()

    def GetTimeSec(self, relative_to_start_bool=False):
        if relative_to_start_bool:
            return self.absolute_time_sslo_ - self.absolute_time_sslo_[0]
        else:
            # Seconds Since Liftoff(sslo)
            return self.absolute_time_sslo_

    def GetPositionECF3M(self, interp_times_sec, relative_to_start_bool=False):
        spl = CubicSpline(self.GetTimeSec(relative_to_start_bool), self.pos_ecf_3m_, extrapolate=False)
        return spl(interp_times_sec)

    def GetVelocityECF3MPS(self, interp_times_sec, relative_to_start_bool=False):
        spl = CubicSpline(self.GetTimeSec(relative_to_start_bool), self.vel_ecf_3mps_, extrapolate=False)
        return spl(interp_times_sec)

    def GetAccelerationECF3MPS(self, interp_times_sec, relative_to_start_bool=False):
        spl = CubicSpline(self.GetTimeSec(relative_to_start_bool), self.acc_ecf_3mpss_, extrapolate=False)
        return spl(interp_times_sec)

    def GetRPYECF3Deg(self, interp_times_sec, relative_to_start_bool=False):
        spl = CubicSpline(self.GetTimeSec(relative_to_start_bool), self.rpy_ecf_extrinsic_3deg_, extrapolate=False)
        return spl(interp_times_sec)


    def PlotTSP(self):
        const = Constants()
        lla_ddm = Coordtrans.ECF2LLA(const, self.pos_ecf_3m_)



        def check_python_installation(package):
            import importlib

            try:
                importlib.import_module(package)
                return True
            except ImportError:
                return False

        if check_python_installation('simplekml'):
            # Create an instance of Kml
            kml = simplekml.Kml(open=1)
            # Create a linestring that will be extended to the ground but sloped from the ground up to 100m
            linestring = kml.newlinestring(name="A Sloped Line")
            pt_list = []
            for xx in range(lla_ddm.shape[0]):
                pt_list.append((lla_ddm[xx, 1], lla_ddm[xx, 0], lla_ddm[xx, 2]))
            linestring.coords = pt_list
            linestring.altitudemode = simplekml.AltitudeMode.relativetoground
            # linestring.extrude = 1
            # Save the KML
            kml.save(os.path.basename(self.object_name_).split('/')[-1] + ".kml")




        ax = plt.figure().add_subplot(projection='3d')
        rng = np.arange(0, self.pos_ecf_3m_.shape[0], 4)
        delta_time = np.append(np.diff(self.GetTimeSec()[rng]), 0)
        ax.scatter(self.pos_ecf_3m_[rng, 0], self.pos_ecf_3m_[rng, 1], zs=self.pos_ecf_3m_[rng, 2], label='Position')
        ax.quiver(self.pos_ecf_3m_[rng, 0], self.pos_ecf_3m_[rng, 1], self.pos_ecf_3m_[rng, 2]
                  , self.vel_ecf_3mps_[rng, 0] * delta_time, self.vel_ecf_3mps_[rng, 1] * delta_time, self.vel_ecf_3mps_[rng, 2] * delta_time
                  , color='r'
                  , label="velocity vector")

        body_xyz = Rotations.RPY2Body(self.rpy_ecf_extrinsic_3deg_)
        ax.quiver(self.pos_ecf_3m_[rng, 0], self.pos_ecf_3m_[rng, 1], self.pos_ecf_3m_[rng, 2]
                  , body_xyz[rng, 0] / np.linalg.norm(body_xyz[rng, :], axis=1) * np.linalg.norm(self.vel_ecf_3mps_[rng, :], axis=1) * delta_time
                  , body_xyz[rng, 1] / np.linalg.norm(body_xyz[rng, :], axis=1) * np.linalg.norm(self.vel_ecf_3mps_[rng, :], axis=1) * delta_time
                  , body_xyz[rng, 2] / np.linalg.norm(body_xyz[rng, :], axis=1) * np.linalg.norm(self.vel_ecf_3mps_[rng, :], axis=1) * delta_time
                  , color='k'
                  , label="body vector")
        ax.legend()


        lat = np.linspace(np.min(lla_ddm[:, 0]), np.max(lla_ddm[:, 0]), 100)
        lon = np.linspace(np.min(lla_ddm[:, 1]), np.max(lla_ddm[:, 1]), 100)
        surf_ecf = np.zeros((lat.shape[0], lon.shape[0], 3))
        for ii in range(len(lat)):
            surf_ecf[ii, :, :] = Coordtrans.LLA2ECF(const, np.vstack([np.ones(len(lon)) * lat[ii], lon, np.zeros(len(lon))]).transpose())
        ax.plot_surface(surf_ecf[:, :, 0], surf_ecf[:, :, 1], surf_ecf[:, :, 2], cmap='Greens', alpha=.25)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        plt.figure()
        plt.plot(lla_ddm[:, 1], lla_ddm[:, 0])
        plt.xlabel('longitude')
        plt.ylabel('latitutde')
        plt.title("Longitude vs Altitude for a TSP aircraft")

        fig_p, axs_pos = plt.subplots(3)
        # fig_all, axs_all = plt.subplots(3)
        fig_v, axs_vel = plt.subplots(3)
        fig_p.suptitle('Position')
        fig_v.suptitle('Velocity')

        # diff_time = np.diff(self.GetTimeSec())
        for ii in range(3):
            delta_pos_vel = np.diff(self.pos_ecf_3m_[:, ii]) / np.diff(self.GetTimeSec())
            outliers = np.logical_and(delta_pos_vel < np.mean(delta_pos_vel) - 3 * np.std(delta_pos_vel), np.mean(delta_pos_vel) + 3 * np.std(delta_pos_vel) < delta_pos_vel)
            delta_pos_vel[outliers] = 0

            axs_pos[ii].plot(self.GetTimeSec(), self.pos_ecf_3m_[:, ii], label='Diff Pos')
            axs_pos[ii].grid()
            axs_pos[ii].set_ylabel("Dist[m]")

            axs_vel[ii].plot(self.GetTimeSec()[1:], delta_pos_vel, label='Diff Pos')
            axs_vel[ii].plot(self.GetTimeSec(), self.vel_ecf_3mps_[:, ii], '--', label='Velocity Measured')
            axs_vel[ii].grid()
            axs_vel[ii].set_ylabel("Vel[mps]")

            # axs_all[ii].plot(self.GetTimeSec(), -1 * body_xyz[:, ii] - self.vel_ecf_3mps_[:, ii])

            # axs_all[ii].scatter(self.GetTimeSec(), self.pos_ecf_3m_[:, ii], label='Diff Pos')
            # for xx in range(len(diff_time)):
            #     axs_all[ii].plot(np.array((self.GetTimeSec()[xx], self.GetTimeSec()[xx+1]))
            #              , np.array((self.pos_ecf_3m_[xx, ii], self.pos_ecf_3m_[xx, ii] + body_xyz[xx, ii] * np.linalg.norm(self.vel_ecf_3mps_[xx, ii]) / np.linalg.norm(self.vel_ecf_3mps_[xx, ii]) * diff_time[xx])))
            #     axs_all[ii].plot(np.array((self.GetTimeSec()[ii], self.GetTimeSec()[ii + 1]))
            #              , np.array((self.pos_ecf_3m_[xx, ii], self.pos_ecf_3m_[xx, ii] + self.vel_ecf_3mps_[xx, ii] * diff_time[xx])))

        axs_pos[2].set_xlabel("time[sec]")
        axs_vel[2].set_xlabel("time[sec]")
        plt.legend()

        

