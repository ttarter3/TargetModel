import os

from partyfirst.targetmodel.include.WavefrontObjFile import WavefrontObjFile


# conda did not work when trying to install tinyobjloader
# only thing that worked was downloading microsoft build tools -> https://visualstudio.microsoft.com/visual-cpp-build-tools/ -> Download build tools -> run exe -> next -> next -> next ...
# Then running ".\pip install tinyobjloader==2.0.0rc7" which contains precompiled windows libraries
# newer versions of tinyobjloader did not work. i.e. 2.0.0.rc9, idk about rc8
# import tinyobjloader
# tinyobjloader and pyassimp did not work.  just parse it ourselves.

class Model3D:

    def __init__(self, object_name: str
                 , object_wavefront_obj_file: str) -> object:
        """
        Generic Model for hosting a 3d Object
        :param object_name: UniqueId for the object of interest. Multiple objects of similar builds would have 'airplane1', 'airplane2'
        """
        self.object_name_ = object_name
        self.wavefront_obj_ = WavefrontObjFile(object_wavefront_obj_file)

if __name__ == "__main__":
    phantom_obj_file = os.path.join(
                        os.path.dirname(os.path.realpath(__file__))
                        , '..', 'data', 'DJIPantomSmoothed.obj')
    meshes = Model3D('DJI_Phantom', phantom_obj_file)
    meshes.wavefront_obj_.PlotVertices()
