from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='set_nms_module',
      ext_modules=[cpp_extension.CppExtension('set_nms_module', ['set_nms_module.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})