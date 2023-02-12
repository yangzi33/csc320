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
import numpy as np
import pandas as pd

import viscomp
import viscomp.ops.image as img_ops

def load_csv_path(path):
    """Loads a path to a csv file with a numpy matrix.

    Args:
        path (str): Path to a CSV file.
    """
    if not os.path.exists(path):
        raise Exception(f"Path {path} does not exist!")
    arr = pd.read_csv(path, header=None).to_numpy()
    return arr

if __name__ == '__main__':
    # Parser to load command line arguments.
    parser = viscomp.parse_options()
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--image-path', type=str, required=True, 
                           help='Path to the image to use for the assignment')
    app_group.add_argument('--source-path', type=str, required=True,
                           help='Path to the CSV file containing the source coords')
    app_group.add_argument('--destination-path', type=str, required=True,
                           help='Path to the CSV file containing the destination coords')
    app_group.add_argument('--output-path', type=str, required=True,
                           help='Path to output the resulting images to')
    app_group.add_argument('--homography-path', type=str,
                           help='(Optional) Path to a CSV file containing a homography matrix. '
                                 'If set, will use this instead of calculating one from coords.')
    app_group.add_argument('--reference-solution', action='store_true',
                           help='Set this to use the precompiled binary to run the reference solution.') 

    args = parser.parse_args()
    
    import sourcedefender
    if args.reference_solution:
        import viscomp.binaries as algos
    else:
        import viscomp.algos as algos
    
    # Load the data paths passed in through command line arguments
    image = img_ops.read_image(args.image_path)
    h, w, _ = image.shape

    print(f"Running algorithm on {os.path.abspath(args.image_path)}...")
    source_coords = load_csv_path(args.source_path)
    destination_coords = load_csv_path(args.destination_path)
    source_coords = img_ops.normalize_coordinates(source_coords, h, w)
    destination_coords = img_ops.normalize_coordinates(destination_coords, h, w)
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]

    np.set_printoptions(formatter={'float': '{:.4f}'.format})
    print("")
    print(f"Source Coords in Normalized [-1,1] Coordinates:")
    print(source_coords)

    print("")
    print(f"Destination Coords in Normalized [-1,1] Coordinates:")
    print(destination_coords)

    preset_homography = None
    if args.homography_path is not None:
        preset_homography = load_csv_path(args.homography_path)
        homography = preset_homography
    else:
        homography = algos.calculate_homography(source_coords, destination_coords)
    points = homography @ np.concatenate([destination_coords, np.ones([4, 1])], axis=-1).T
    backprojected_coords = (points.T)[:4, :2] / (points.T)[:4, 2:]
    print("")
    print(f"Destination Coords Backprojected (should match source!)")
    print(backprojected_coords)
    print("")

    # IMPORTANT: Flip the y-axis of the image to set it to be in the correct coordinate space for this assignment.
    image = np.flip(image, 0)

    # Run the algorithm. You will implement this algorithm and is implemented inside `viscomp/algos/a1.py`.
    output = algos.run_a1_algo(image, np.zeros_like(image), source_coords, destination_coords, 
                               homography=preset_homography)
    output = np.flip(output, 0)

    # Write the output to image.
    output_path = os.path.abspath(os.path.join(args.output_path, image_name + "_output.png"))
    print(f"Algorithm success. Writing output image to {output_path}.")
    img_ops.write_image(output, output_path)
