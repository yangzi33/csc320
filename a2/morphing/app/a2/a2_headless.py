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
import numpy as np
import pandas as pd
import cv2
import tqdm
import viscomp
import viscomp.ops.image as img_ops
import viscomp.ops.csv as csv_ops

if __name__ == '__main__':
    parser = viscomp.parse_options()
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--source-image-path', type=str, required=True,
                           help='Path to the source image to use for the assignment.')
    app_group.add_argument('--destination-image-path', type=str,
                           help='Path to the destination image to use for the assignment.'
                                ' If this image is provided, the app will enter interpolation mode where the'
                                ' goal is to interpolate between the two images.')
    app_group.add_argument('--verbose', action='store_true',
                           help='Set this to print out more debug outputs')
    app_group.add_argument('--param-a', type=float, default=1e-3,
                           help='The a parameter from the Beier-Neely paper.')
    app_group.add_argument('--param-b', type=float, default=1.6,
                           help='The b parameter from the Beier-Neely paper.')
    app_group.add_argument('--param-p', type=float, default=0.5,
                           help='The p parameter from the Beier-Neely paper.')
    app_group.add_argument('--param-t', type=float, default=1.0,
                           help='The t parameter for interpolating between two images.')
    app_group.add_argument('--output-path', type=str, default="results",
                           help='Path to folder to export output image and csv files for the morph lines drawn on the images.')
    app_group.add_argument('--source-line-path', type=str, required=True,
                           help='Path to csv file to import preset left morph lines.') 
    app_group.add_argument('--destination-line-path', type=str, required=True,
                           help='Path to csv file to import preset right morph lines.') 
    app_group.add_argument('--supersampling', type=int, default=1,
                           help='The patch size for supersampling.')
    app_group.add_argument('--vectorize', action='store_true',
                           help='Set this to enable vectorization.') 
    app_group.add_argument('--reference-solution', action='store_true',
                           help='Set this to use the precompiled binary to run the reference solution.') 
    app_group.add_argument('--bilinear', action='store_true',
                           help='Set this to enable bilinear interpolation.') 
    app_group.add_argument('--debug-grid', action='store_true',
                           help='Set this to draw a grid to debug morphing.') 
    app_group.add_argument('--num-frames', type=int, default=1,
                           help='The number of frames to output for interpolation.')

    args = parser.parse_args()
    if args.reference_solution:
        import sourcedefender
        import viscomp.binaries as algos
    else:
        import viscomp.algos as algos
    
    source_image = np.flip(img_ops.read_image(args.source_image_path), 0)
    if not args.destination_image_path:
        destination_image = np.zeros_like(source_image)
        print(f"Running algorithm on {os.path.abspath(args.source_image_path)}...")
    else:
        destination_image = np.flip(img_ops.read_image(args.destination_image_path), 0)
        if destination_image.shape[0] != source_image.shape[0] or \
           destination_image.shape[1] != destination_image.shape[1]:
            print("The source image dimensions should match the destination image dimensions."
                            f" Got {source_image.shape} and {destination_image.shape}.")
            sys.exit(0)
        print(f"Running algorithm on {os.path.abspath(args.source_image_path)} "
              f"and {os.path.abspath(args.destination_image_path)}...")
    

    if args.debug_grid:
        hi, wi = source_image.shape[:2]
        for i in range(10):
            source_image[i*(hi//10), :] = 0
            source_image[:, i*(wi//10)] = 0
        hi, wi = destination_image.shape[:2]
        for i in range(10):
            destination_image[i*(hi//10), :] = 0
            destination_image[:, i*(wi//10)] = 0

    source_morph_lines = csv_ops.load_csv(args.source_line_path).reshape(-1, 2, 2)
    source_morph_lines = img_ops.normalize_coordinates(source_morph_lines, *source_image.shape[:2])
    
    destination_morph_lines = csv_ops.load_csv(args.destination_line_path).reshape(-1, 2, 2)
    destination_morph_lines = img_ops.normalize_coordinates(destination_morph_lines, *source_image.shape[:2])
    
    source_image_name = os.path.splitext(os.path.basename(args.source_image_path))[0]
    
    if args.destination_image_path:
        destination_image_name = os.path.splitext(os.path.basename(args.destination_image_path))[0]
        source_image_name += "_" + destination_image_name

        
    if args.num_frames == 1:
        frame_ts = np.array([1.0])
    else:
        frame_ts = np.arange(0, args.num_frames+1).astype(np.float32) / args.num_frames
        
    print("")
    for i, t in enumerate(list(frame_ts)):    
        if args.num_frames > 1:
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            percent_done = float(i)/float(frame_ts.shape[0]-1)
            print(f"[{'#' * int(percent_done*30)}{'-' * (30-int(percent_done*30))}] {int(100*percent_done)}% done")
        warped_source = algos.run_a2_algo(source_image, destination_image, 
                                         source_morph_lines, destination_morph_lines,
                                         args.param_a, args.param_b, args.param_p,
                                         args.supersampling, args.bilinear, 
                                         False, t, args.vectorize)
        warped_source[..., -1] = 1.0
        img_ops.write_image(np.flip(warped_source, 0), os.path.join(args.output_path, f'{source_image_name}_warpedsource_{i:06d}.png'))
        if args.destination_image_path:
            warped_destination = algos.run_a2_algo(destination_image, source_image, 
                                             destination_morph_lines, source_morph_lines,
                                             args.param_a, args.param_b, args.param_p,
                                             args.supersampling, args.bilinear, 
                                             False, 1.0 - t, args.vectorize)
            
            warped_destination[..., -1] = 1.0
            morphed_image = warped_source * (1.0 - t) + warped_destination * t
            morphed_image[..., -1] = 1.0
            img_ops.write_image(np.flip(warped_destination, 0), os.path.join(args.output_path, f'{source_image_name}_warpeddestination_{i:06d}.png'))
            img_ops.write_image(np.flip(morphed_image, 0), os.path.join(args.output_path, f'{source_image_name}_morphed_{i:06d}.png'))
