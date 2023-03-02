# CSC320 Spring 2023
# Assignment 2
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
import glob

paths = sorted(glob.glob("data/f/*"))
for path in paths:
    if os.path.isdir(path):
        img_name = os.path.splitext(os.path.basename(path))[0]
        dirname = os.path.basename(path)
        cmd = f"python a2_headless.py --source-image-path data/f/f.png \
                --source-line-path {path}/source.csv \
                --destination-line-path {path}/destination.csv --output-path test_results/{dirname}"
        for arg in sys.argv[1:]:
            cmd += " " + arg
        os.system(cmd)
