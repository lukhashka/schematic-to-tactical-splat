from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='splat_generator_core',
    ext_modules=[
        # Змінюємо ім'я модуля на src.splat_generator_core
        CppExtension('src.splat_generator_core', ['src/splat_generator_core.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)