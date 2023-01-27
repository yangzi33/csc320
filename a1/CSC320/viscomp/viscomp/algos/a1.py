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

import sys
import numpy as np
import cv2
import viscomp.ops.image as img_ops

def run_a1_algo(source_image, destination_image, source_coords, destination_coords, homography=None):
    """Run the entire A1 algorithm.

    Args: 
        source_image (np.ndarray): The source image of shape [Hs, Ws, 4]
        destination_image (np.ndarray): The destination image of shape [Hd, Wd, 4]
        source_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the source image.
        destination_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the destination image.
        homography (np.ndarray): (Optional) [3, 3] homography matrix. If passed in, will use this
                                 instead of calculating it.
    
    Returns:
        (np.ndarray): Written out image of shape [Hd, Wd, 4]
    """
    if homography is None:
        print("Calculating homography...")
        np.set_printoptions(formatter={'float': '{:.4f}'.format})
        homography = calculate_homography(source_coords, destination_coords)
    else:
        print("Using preset homography matrix...")
    print("")
    print("Homography matrix:")
    print(homography)
    print("")
    print("Performing backward mapping...")
    output_buffer = backward_mapping(homography, source_image, destination_image, destination_coords)
    print("Algorithm has succesfully finished running!")
    return output_buffer

def convex_polygon(poly_coords, image_coords):
    """From coords that define a convex hull, find which image coordinates are inside the hull.

     Args:
         poly_coords (np.ndarray): [N, 2] list of 2D coordinates that define a convex polygon.
                              Each nth index point is connected to the (n-1)th and (n+1)th 
                              point, and the connectivity wraps around (i.e. the first and last
                              points are connected to each other)
         image_coords (np.ndarray): [H, W, 2] array of coordinates on the image. Using this, 
                                 the goal is to find which of these coordinates are inside
                                 the convex hull of the polygon.
         Returns:
             (np.ndarray): [H, W] boolean mask where True means the coords is inside the hull.
    """
    mask = np.ones_like(image_coords[..., 0]).astype(np.bool)
    N = poly_coords.shape[0]
    for i in range(N):
        dv = poly_coords[(i+1)%N] - poly_coords[i]
        winding = (image_coords - poly_coords[i][None]) * (np.flip(dv[None], axis=-1))
        winding = winding[...,0] - winding[...,1]
        mask = np.logical_and(mask, (winding > 0))
    return mask

# student_implementation

def calculate_homography(source, destination):
    """Calculate the homography matrix based on source and desination coordinates.

    Args:
        source (np.ndarray): [4, 2] matrix of 2D coordinates in the source image.
        destination (np.ndarray): [4, 2] matrix of 2D coordinates in the destination image.

    Returns:
        (np.ndarray): [3, 3] homography matrix.
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    # p source
    # p' destination
    # homography = np.eye(3)

    x1, x2, x3, x4 = source[0, 0], source[1, 0], source[2, 0], source[3, 0]
    y1, y2, y3, y4 = source[0, 1], source[1, 1], source[2, 1], source[3, 1]

    xp1, xp2, xp3, xp4 = destination[0, 0], destination[1, 0], destination[2, 0], destination[3, 0]
    yp1, ypp2, yp3, yp4 = destination[0, 1], destination[1, 1], destination[2, 1], destination[3, 1]

    # Flatten vector equation into form of Ax = b
    A = np.array([
        [x1, y1, 1, 0, 0, 0, -x1*xp1, -y1*xp1, -xp1],
        [0, 0, 0, x1, y1, 1, -x1*yp1, -y1*yp1, -yp1],
        [x2, y2, 1, 0, 0, 0, -x2*xp2, -y2*xp2, -xp2],
        [0, 0, 0, x2, y2, 1, -x2*ypp2, -y2*ypp2, -ypp2],
        [x3, y3, 1, 0, 0, 0, -x3*xp3, -y3*xp3, -xp3],
        [0, 0, 0, x3, y3, 1, -x3*yp3, -y3*yp3, -yp3],
        [x4, y4, 1, 0, 0, 0, -x4*xp4, -y4*xp4, -xp4],
        [0, 0, 0, x4, y4, 1, -x4*yp4, -y4*yp4, -yp4],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

    homography = np.solve(A, b).reshape(3, 3)
    #################################
    ######### DO NOT MODIFY #########
    #################################
    return homography

def backward_mapping(transform, source_image, destination_image, destination_coords):
    """Perform backward mapping onto the destination image.

    The goal of this function is to map each destination image pixel which is within the polygon defined
    by destination_coords to a corresponding image pixel in source_image.

    Hints: Start by iterating through the destination image pixels using a nested for loop. For each pixel,
    use the convex_polygon function to find whether they are inside the polygon. If they are, figure out 
    how to use the homography matrix to find the corresponding pixel in source_image.

    Args:
        transform (np.ndarray): [3, 3] homogeneous transformation matrix.
        source_image (np.ndarray): The source image of shape [Hs, Ws, 4]
        destination_image (np.ndarray): The destination image of shape [Hd, Wd, 4]
        source_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the source image.
        destination_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the destination image.
     
    Returns:
        (np.ndarray): [Hd, Wd, 4] image with the source image projected onto the destination image.
    """
    h, w, _ = destination_image.shape
    output_buffer = np.zeros_like(destination_image)
    
    # The integer coordinates which you can access via xs_int[r, c]
    xs_int = img_ops.create_coordinates(h, w)
    
    # The float coordinates [-1, 1] which you can access via xs[r, c]
    # To avoid confusion, you should always denormalize this using img_ops.denormalize_coordinates(xs, h, w)
    # which will bring it back to pixel space, and avoid doing any pixel related operations (like filtering,
    # interpolation, etc) in normalized space. Normalized space however is nice for heavy numerical operations
    # for floating point precision reasons.
    xs = img_ops.normalize_coordinates(xs_int, h, w) 
    
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################

    # One way you can implement this is with a double for loop, like the following.
    # You DO NOT necessarily need to implement it in this way... you can implement
    # this entire assignment pretty easily by utilizing vectorization.
    # As of matter fact, I (the TA) personally think that the vectorized version of the
    # code is simpler and less lines of code than the double for loop version.
    # That being said if you still don't find vectorization natural, go ahead and attempt
    # the double for loop solution!
    for r in range(h):
        # The double for loop is slow, so we implement a progress bar.a
        # tqdm (a progress bar library) doesn't work great with certain GUI libraries, 
        # so we implment our own progress bar here.
        # you should ignore this code for the most part.
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
        percent_done = float(r)/float(h-1)
        print(f"[{'#' * int(percent_done*30)}{'-' * (30-int(percent_done*30))}] {int(100*percent_done)}% done")
        
        for c in range(w):
            pixel_coord = xs[r, c]
            # Do stuff here!

    #################################
    ######### DO NOT MODIFY #########
    #################################
    return output_buffer
