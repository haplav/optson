from setuptools import setup

import versioneer

# Setup configuration
setup(
    package_data={"optson": ["py.typed"]},
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
