import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.linalg import norm

from config.Config import Config
from partyfirst.Broker.MessageInterface import MessageInterface
from partyfirst.GUID import GUID
from partyfirst.radarmodel.include.panel.Panel import Panel
from partyfirst.targetmodel.include.RCS.RCS import RCS
from partyfirst.targetmodel.include.RCS.RCSProfile import RCSProfile
from partyfirst.targetmodel.include.TSP.TSP import TSP
from partythird.constants.include.Constants import Constants
from partythird.coordtrans.include.coordtrans import Rotations, Coordtrans

# ToDo: Need to add the ability to generate RCS model based on waveform.  I.e. are we s band waveform or x band waveform

class Target(object):
  def __init__(self, tsp: TSP, rcs_model: RCS):
    super().__init__()
    self.guid_ = GUID.generate_guid()

    assert(isinstance(rcs_model, RCS) and isinstance(tsp, TSP))

    self.tsp_ = tsp
    self.rcs_model_ = rcs_model

  def on_failure(self):
    pass
  def on_receive(self, msg):
    pass

  def GetRCS(self, time_sec: np.array, tx_pos_ecf_3m, rx_pos_ecf_3m) -> RCSProfile:
    # [p, v, a] = Coordtrans.ECF2RAE(Constants(), relative_reference_position_lla
    #                    , self.tsp_.GetPositionECF3M(time_sec)
    #                    , self.tsp_.GetVelocityECF3MPS(time_sec)
    #                    , self.tsp_.GetAccelerationECF3MPS(time_sec))

    tgt_pos_ecf_3m = self.tsp_.GetPositionECF3M(time_sec)
    tgt_rpy_ecf_3d = self.tsp_.GetRPYECF3Deg(time_sec)

    med_tgt_pos_3m  = np.expand_dims(np.median(tgt_pos_ecf_3m, axis=0), axis=0)
    relative_reference_position_lla = Coordtrans.ECF2LLA(Constants(), tx_pos_ecf_3m)
    med_tgt_r_m = Coordtrans.ECF2RAE(Constants(), relative_reference_position_lla, med_tgt_pos_3m)[0, 0]

    assert(np.isnan(med_tgt_r_m) or med_tgt_r_m >= 0)

    obj_reference_vec = tx_pos_ecf_3m - tgt_pos_ecf_3m
    # obj_reference_vec = np.divide(obj_reference_vec.T, np.linalg.norm(obj_reference_vec, axis=1)).T
    obj_reference_rae = Coordtrans.ENU2RAE(obj_reference_vec)

    # ToDo: Verify that we should be using RPY instead of Velocity
    # obj_facing_vec = Rotations.RPY2Body(tgt_rpy_ecf_3d)
    obj_facing_vec = self.tsp_.GetVelocityECF3MPS(time_sec)
    # obj_facing_vec = np.divide(obj_facing_vec.T, np.linalg.norm(obj_facing_vec, axis=1)).T
    obj_facing_rae = Coordtrans.ENU2RAE(obj_facing_vec)

    delta_frames = obj_facing_rae - obj_reference_rae

    roll_deg = delta_frames[:, 2]
    yaw_deg = delta_frames[:, 1]

    rcs_profile = self.rcs_model_.GetRCS(roll_deg, yaw_deg)

    tgt_in_rng_window_time = med_tgt_r_m / Constants.C() * 2 + rcs_profile.time_vector_reference_relative_ - rcs_profile.reference_time_2_tgt_range_

    # min_tgt = min(time_sec) < med_tgt_r_m / Constants.C() * 2
    # max_tgt = med_tgt_r_m / Constants.C() * 2 < max(time_sec)
    # tgt_in_rng_window = np.where(np.logical_and(min(time_sec) < tgt_in_rng_window_time, tgt_in_rng_window_time <= max(time_sec)))
    # samples_in_rcs_window = np.where(np.logical_and(min(tgt_in_rng_window_time) < time_sec, time_sec <= max(tgt_in_rng_window_time)))

    if Config.config("plot")["tgt_rcs_profile_in_rng_window_comp"]:
      plt.figure()
      plt.scatter(time_sec, np.zeros_like(time_sec), label=f"Rng Window: t between samp => {Constants.C() * (time_sec[1] - time_sec[0]):.3f} m")
      plt.scatter(tgt_in_rng_window_time, np.ones_like(tgt_in_rng_window_time), label=f"RCS Profile: tgt size => {Constants.C() * (tgt_in_rng_window_time[-1] - tgt_in_rng_window_time[0]):.3f} m")
      plt.title("Target RCS Profile in Range Window Comparison")
      plt.legend()

    # additive_rcs_data = np.interp(time_sec, tgt_in_rng_window_time, rcs_profile.rcs_profile_square_meters_, right=0, left=0)

    group_idx = np.digitize(tgt_in_rng_window_time, time_sec)
    additive_rcs_data = np.zeros_like(time_sec)
    for ii in range(min(group_idx), max(group_idx)):
      additive_rcs_data[ii] = np.max(rcs_profile.rcs_profile_square_meters_[group_idx == ii])

    # sum(abs(additive_rcs_data))

    return additive_rcs_data

  def GetTime2TargetRelative(self, time_of_transmit, tx_pos_ecf_3m):
    # Calculate the distance to all points of target

    if np.max(self.tsp_.absolute_time_sslo_) < time_of_transmit:
      return np.inf

    tgt_time_sec = self.tsp_.absolute_time_sslo_
    tgt_pos_ecf_3m = self.tsp_.pos_ecf_3m_
    pos_delta = tgt_pos_ecf_3m - tx_pos_ecf_3m
    unit_vector_2_target = pos_delta / np.linalg.norm(pos_delta)
    propogation_time = tgt_time_sec - time_of_transmit

    propogation_range = norm(unit_vector_2_target, axis=1) * propogation_time * Constants.C()

    idx_signal_passes_tgt = np.where(
        norm(pos_delta, axis=1) < propogation_range
    )[0]

    if len(idx_signal_passes_tgt) == 0:
      return np.inf

    idx_signal_passes_tgt = idx_signal_passes_tgt[0]
    if idx_signal_passes_tgt == 0:
      return np.inf

    time_2_target_sec = norm(pos_delta[idx_signal_passes_tgt, :]) / Constants.C()

    # ToDo: Add a interpolation to get a higher accuracy range if the object is not inheritently high fidelity.  The polyfit below will need to be refactor but could work
    # # Perform second-order polynomial fit (quadratic fit)
    # step = 3
    # valid_idx = np.arange(np.max((0, intercept_index - step)), np.min((rng_2_tgt.shape[0] - 1, intercept_index + step)))
    #
    # coefficients = np.polyfit(tgt_time_sec[valid_idx], rng_2_tgt[valid_idx], 2)
    # # Generate polynomial from the coefficients
    # polynomial = np.poly1d(coefficients)
    # # Generate x values for plotting the fitted curve
    # x_fit = np.linspace(np.min(tgt_time_sec[valid_idx]), np.max(tgt_time_sec[valid_idx]), 1000)
    # y_fit = polynomial(x_fit)
    #
    # intercept_index = np.argmin(y_fit)


    return time_2_target_sec

    # signal_alive_range = Constants.C() * signal_alive_time
    #
    # plane_wave_range_from_radar = tx_pos_ecf_3m + np.multiply(unit_vector_2_target, signal_alive_range[:, np.newaxis])
    #
    #
    #
    # rng_2_tgt = np.linalg.norm(plane_wave_range_from_radar - tgt_pos_ecf_3m, axis=1)
    #
    # intercept_index = np.argmin(rng_2_tgt)
    # if intercept_index == 0 or intercept_index == len(tgt_time_sec):
    #   return np.inf
    #
    # # Perform second-order polynomial fit (quadratic fit)
    # step = 3
    # valid_idx = np.arange(np.max((0, intercept_index - step)), np.min((rng_2_tgt.shape[0] - 1, intercept_index + step)))
    #
    # coefficients = np.polyfit(tgt_time_sec[valid_idx], rng_2_tgt[valid_idx], 2)
    # # Generate polynomial from the coefficients
    # polynomial = np.poly1d(coefficients)
    # # Generate x values for plotting the fitted curve
    # x_fit = np.linspace(np.min(tgt_time_sec[valid_idx]), np.max(tgt_time_sec[valid_idx]), 1000)
    # y_fit = polynomial(x_fit)
    #
    # intercept_index = np.argmin(y_fit)
    #
    # next_idx = np.where(tgt_time_sec > time_of_transmit)
    # if len(next_idx) > 0 and len(next_idx[0]) > 0:
    #   intercept_index = max(intercept_index, next_idx[0][0])
    #
    # tgt_time_tmp_sec = x_fit[intercept_index]
    #
    #

    # if Config.config("plot")["tgt_2_signal_propogation"]:
    #   plt.figure()
    #   plt.plot(x_fit, y_fit)
    #   plt.scatter(x_fit[intercept_index], y_fit[intercept_index])
    #   plt.title("esitimate of the time to target.  We are currently expecting all values to be a generalize concave up parabola")
    #
    # # ToDo: this should really be a interpolation instead of picking the closest neighbor.  Find closest then interpolate to higher order quadratic
    # time_2_target = tgt_time_tmp_sec - time_of_transmit
    #
    # assert(time_2_target >= 0) # If a wave propagates, the time_2_target should always be positive

    # return time_2_target


  def GetRCSAtReceiver(self
                       , time_of_transmit: float
                       , time_sec: np.array
                       , tx_panel: Panel
                       , rx_panel: Panel) -> RCSProfile:

    time_2_target_sec = self.GetTime2TargetRelative(time_of_transmit, tx_panel.ref_ecf_pos_3m_)
    return self.GetRCS(time_sec - time_2_target_sec, tx_panel.ref_ecf_pos_3m_, rx_panel.ref_ecf_pos_3m_)

if __name__ == '__main__':
  import numpy as np

#
#   def vectors_to_angles(vector_a, vector_b):
#     # Normalize the vectors to ensure they are unit vectors
#     vector_a = vector_a / np.linalg.norm(vector_a)
#     vector_b = vector_b / np.linalg.norm(vector_b)
#
#     # Calculate the azimuth angle (rotation about the vertical axis)
#     azimuth = np.arctan2(vector_b[1], vector_b[0]) - np.arctan2(vector_a[1], vector_a[0])
#
#     # Ensure the azimuth angle is in the range [-np.pi, np.pi)
#     azimuth = (azimuth + np.pi) % (2 * np.pi) - np.pi
#
#     # Calculate the elevation angle (angle above the horizontal plane)
#     elevation = np.arcsin(vector_b[2]) - np.arcsin(vector_a[2])
#
#     # Convert angles to degrees for better readability
#     azimuth_degrees = np.degrees(azimuth)
#     elevation_degrees = np.degrees(elevation)
#
#     return azimuth_degrees, elevation_degrees
#
#
#   # Example vectors
#   vector_a = np.array([1, 1, 1])
#   vector_b = np.array([0, -1, 0])
#
#   vector_a = vector_a / np.linalg.norm(vector_a)
#   vector_b = vector_b / np.linalg.norm(vector_b)
#
#   # Find azimuth and elevation angles
#   azimuth_angle, elevation_angle = vectors_to_angles(vector_a, vector_b)
#
#   # Print the results
#   print("Azimuth:", azimuth_angle)
#   print("Elevation:", elevation_angle)
#
#   r = scipy.spatial.transform.Rotation.from_euler('xz', [elevation_angle, azimuth_angle], degrees=True)
#   r = vector_a @ r.as_matrix()
#   r = r / np.linalg.norm(r)
#
#   print(f"Vec  A: {vector_a}")
#   print(f"RotAxR: {r}")
#   print(f"Vec  B: {vector_b}")
#   print(f"ERROR: {np.linalg.norm(vector_b - r)}")
#
#   print("APPEARS TO BE WORKING")
#
# '''
#   import numpy as np
#
#   def vectors_to_angles(vector_a, vector_b):
#       # Normalize the vectors to ensure they are unit vectors
#       vector_a = vector_a / np.linalg.norm(vector_a)
#       vector_b = vector_b / np.linalg.norm(vector_b)
#
#       # Calculate the dot product and cross product of the two vectors
#       dot_product = np.dot(vector_a, vector_b)
#       cross_product = np.cross(vector_a, vector_b)
#
#       # Calculate the yaw angle (rotation about the vertical axis)
#       yaw = np.arctan2(cross_product[0], dot_product)
#
#       # Calculate the roll angle (rotation about the forward axis)
#       roll = np.arctan2(cross_product[1], cross_product[2])
#
#       # Convert angles to degrees for better readability
#       roll_degrees = np.degrees(roll)
#       yaw_degrees = np.degrees(yaw)
#
#       return roll_degrees, yaw_degrees
#
#   # Example vectors
#   vector_a = np.array([1, 0, 1])
#   vector_b = np.array([-1, 0, 1])
#
#   # Find roll and yaw angles
#   roll_angle, yaw_angle = vectors_to_angles(vector_a, vector_b)
#
#   # Print the results
#   print("Roll:", roll_angle)
#   print("Yaw:", yaw_angle)
#
#   r = scipy.spatial.transform.Rotation.from_euler('xz', [roll_angle, yaw_angle], degrees=True)
#   r = vector_a @ r.as_matrix()
#   print(f"ERROR: {np.linalg.norm(vector_b - r)}")
# '''