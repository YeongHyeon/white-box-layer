from setuptools import setup, find_packages

setup(
    name                = 'whiteboxlayer',
    version             = '0.1.6',
    description         = 'TensorFlow based custom layers',
    author              = 'YeongHyeon Park',
    author_email        = 'young200405@gmail.com',
    url                 = 'https://github.com/YeongHyeon/white-box-layer',
    download_url        = 'https://github.com/YeongHyeon/white-box-layer/archive/refs/heads/master.zip',
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
