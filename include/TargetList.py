from partyfirst.targetmodel.include.Target import Target


class TargetList():
  def __init__(self):
    self.targets_ = []

  def AddTarget(self, target: Target):
    assert(isinstance(target, Target) or isinstance(target, TargetList))

    if isinstance(target, TargetList):
      self.targets_.extend(target.targets_)
    else: self.targets_.append(target)