from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cluster_nms_module',
      ext_modules=[cpp_extension.CppExtension('cluster_nms_module', ['cluster_nms_module.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})