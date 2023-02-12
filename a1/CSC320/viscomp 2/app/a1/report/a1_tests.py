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
import glob

paths = sorted(glob.glob("./*.jpg"))
for path in paths:
    img_name = os.path.splitext(os.path.basename(path))[0]
    cmd = f"python a1_headless.py --image-path {path} --source-path tests/{img_name}_source.csv \
            --destination-path tests/{img_name}_destination.csv --output-path test_results"
    os.system(cmd)
