from setuptools import setup
from setuptools_rust import Binding, RustExtension


setup(
    rust_extensions=[
        RustExtension("dl_manager.accelerator", path='dl_manager/accelerator/Cargo.toml', binding=Binding.PyO3)
    ],
    zip_safe=False
)
