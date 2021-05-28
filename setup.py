from setuptools import setup, find_packages

setup(
    name                = 'white-box-layer',
    version             = '0.1.1',
    description         = 'low level tensorflow custom layers',
    author              = 'YeongHyeon Park',
    author_email        = 'young200405@gmail.com',
    url                 = 'https://github.com/YeongHyeon/white-box-layer',
    download_url        = 'https://github.com/YeongHyeon/white-box-layer/archive/refs/heads/master.zip',
    install_requires    = [],
    packages            = find_packages(exclude = []),
    keywords            = ['white-box-layer'],
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
