

class StateInformation:
    def __init__(self, time_sec, ecf_xyz_m, ecf_xyz_d_mps=None, ecf_xyz_dd_mpss=None, ecf_extrinsic_rpy_deg=None):
        self.time_sec_ = time_sec
        self.ecf_xyz_m_ = ecf_xyz_m
        self.ecf_xyz_d_mps_ = ecf_xyz_d_mps
        self.ecf_xyz_dd_mpss_ = ecf_xyz_dd_mpss
        self.ecf_extrinsic_rpy_deg_ = ecf_extrinsic_rpy_deg
