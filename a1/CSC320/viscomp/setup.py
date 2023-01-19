# CSC320 Spring 2023
# Assignment 1
# (c) Kyros Kutulakos, Towaki Takikawa, Esther Lin
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

import os
import sys
from setuptools import setup, find_packages, dist
import glob
import logging
import subprocess

PACKAGE_NAME = 'viscomp'
DESCRIPTION = 'University of Toronto CSC320: viscomp library'
AUTHOR = 'Kyros Kutulakos, Esther Lin, Towaki Takikawa'

if __name__ == '__main__':
    setup(
        # Metadata
        name=PACKAGE_NAME,
        #version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        #url=URL,
        #license=LICENSE,
        python_requires='>=3.8',

        # Package info
        packages=['viscomp'] + find_packages(),
        include_package_data=True,
        zip_safe=True,
        #ext_modules=get_extensions(),
        #cmdclass={
        #    'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)    
        #}

    )
