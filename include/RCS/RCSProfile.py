import numpy as np
from matplotlib import pyplot as plt

from partythird.constants.include.Constants import dB10

class RCSProfile(object):
  def __init__(self, reference_time_2_tgt_range, time_vector_reference_relative, rcs_profile_mm):
    assert (np.squeeze(time_vector_reference_relative).ndim == 1)
    assert (np.squeeze(rcs_profile_mm).ndim == 1)
    assert (isinstance(reference_time_2_tgt_range, float))

    self.time_vector_reference_relative_ = time_vector_reference_relative
    self.rcs_profile_square_meters_ = rcs_profile_mm
    self.reference_time_2_tgt_range_ = reference_time_2_tgt_range


  def PlotRCSProfile(self):
    plt.figure()

    ordinate_rcs_profile = dB10(self.rcs_profile_square_meters_)
    plt.plot(self.time_vector_reference_relative_, ordinate_rcs_profile)
    plt.vlines(self.time_vector_reference_relative_, min(ordinate_rcs_profile), max(ordinate_rcs_profile))