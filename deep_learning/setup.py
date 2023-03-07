from setuptools import setup
from setuptools_rust import Binding, RustExtension

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

setup(
    rust_extensions=[
        RustExtension("dl_manager.accelerator", path='dl_manager/accelerator/Cargo.toml', binding=Binding.PyO3, debug=False)
    ],
    zip_safe=False
)
