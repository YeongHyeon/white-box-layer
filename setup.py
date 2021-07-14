from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name                = 'whiteboxlayer',
    version             = '0.1.14',
    description         = 'TensorFlow based custom layers',
    author              = 'YeongHyeon Park',
    author_email        = 'young200405@gmail.com',
    url                 = 'https://github.com/Orbit-of-Mercury/white-box-layer',
    download_url        = 'https://github.com/Orbit-of-Mercury/white-box-layer/archive/refs/heads/master.zip',
    long_description    = README,
    long_description_content_type   = "text/markdown",
    install_requires    = ['tensorflow==2.3.0'],
    packages            = find_packages(exclude = []),
    keywords            = ['whiteboxlayer'],
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3'
    ],
)
