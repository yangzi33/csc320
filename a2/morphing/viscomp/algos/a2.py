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

import sys
import numpy as np
import cv2
import viscomp.ops.image as img_ops
import math

def run_a2_algo(source_image, destination_image, source_morph_lines, destination_morph_lines,
                param_a, param_b, param_p, supersampling, bilinear, interpolate, param_t, vectorize):
    """Run the entire A2 algorithm.

    In the A2 backward mapping algorithm, there is no linear transformation computed beforehand
    like with A1's homographies. Instead, you directly compute the source_coords based on the 
    Beier-Neely algorithm's formulation.

    For more information about this algorithm (especially with respect to what some of the params
    actually are), consult the Beier-Neely paper provided in class (probably in the course Dropbox).

    Args:
        source_image (np.ndarray): The source image of shape [H, W, 4]
        destination_image (np.ndarray): The destination image of shape [H, W, 4]
        source_morph_lines (np.ndarray): [N, 2] tensor of coordinates for lines in 
                                         normalized [-1, 1] space.
        destination_morph_lines (np.ndarray): [N, 2] tensor of coordinates for lines in
                                              normalized [-1, 1] space.
        param_a (float): The `a` parameter from the Beier-Neely paper controlling morph strength.
        param_b (float): The `b` parameter from the Beier-Neely paper controlling relative
                         line morph strength by distance.
        param_p (float): The `p` parameter from the Beier-Neely paper controlling relative
                         line morph strength by line length.
        supersampling (int): The patch size for supersampling.
        bilinear (bool): If True, will use bilinear interpolation on the pixels.
        interpolate (bool): If True, will interpolate between the two images.
        param_t (float): The interpolation parameter between [0, 1]. If interpolate is False, 
                         will only interpolate the arrows (and not the images).
        vectorize (bool): If True, will use the vectorized version of the code (optional).

    Returns:
        (np.ndarray): Written out image of shape [H, W, 4]
    """
    interpolated_morph_lines = (source_morph_lines * (1.0 - param_t)) + (destination_morph_lines * param_t)
    if interpolate:
        output_image_0 = backward_mapping(source_image, destination_image, 
                source_morph_lines, interpolated_morph_lines,
                param_a, param_b, param_p, supersampling, bilinear, vectorize)
        output_image_1 = backward_mapping(destination_image, source_image, 
                destination_morph_lines, interpolated_morph_lines,
                param_a, param_b, param_p, supersampling, bilinear, vectorize)
        output_buffer = (output_image_0 * (1.0 - param_t)) + (output_image_1 * param_t)
    else:
        output_buffer = backward_mapping(source_image, destination_image, 
                                         source_morph_lines, interpolated_morph_lines,
                                         param_a, param_b, param_p, supersampling, bilinear, vectorize)
    return output_buffer

# student_implementation

def perp(vec):
    """Find the perpendicular vector to the input vector.

    Args:
        vec (np.array): Vectors of size [N, 2].

    Returns:
        (np.array): Perpendicular vectors of size [N, 2].
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    ret = vec.copy()
    ret[:,0], ret[:,1] = ret[:,1], -ret[:,0]
    return ret
    #################################
    ######### DO NOT MODIFY #########
    #################################

def norm(vec):
    """Find the norm (length) of the input vectors.

    Args:
        vec (np.array): Vectors of size [N, 2].

    Returns:
        (np.array): An array of vector norms of shape [N].
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    N = vec.shape[0]
    ans = np.zeros(N)
    for i in range(N):
        ans[i] = np.linalg.norm(vec[i, :])
    return ans
    #################################
    ######### DO NOT MODIFY #########
    #################################


def normalize(vec):
    """Normalize vectors to unit vectors of length 1.

    Hint: Use the norm function you implemented!
    
    Args:
        vec (np.array): Vectors of size [N, 2].

    Returns:
        (np.array): Normalized vectors of size [N, 2].
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    # vec * (1 / l2 norm of vec)
    N = vec.shape[0]
    return vec / norm(vec).reshape((N, 1))
    #################################
    ######### DO NOT MODIFY #########
    #################################

def calculate_uv(p, q, x):
    """Find the u and v coefficients for morphing based on the destination line and destination coordinate.

    This implements Equations 1 and 2 from the Beier-Neely paper.

    This function returns a tuple, which you can expand as follows:

    u, v = calculate_uv(p, q, x)

    Hint #1:
        The functions you implemented above like `norm` and `perp` take in as input a collection
        of vectors, as in size [num_coords, 2] and the like. Often times, you'll find that the arrays 
        you have (like `origin` and `destination`) are size [2]. You can _still_ use these with those
        vectorized functions, by just doing something like `origin[None]` which reshapes the
        array into size [1, 2].

    Args:
        p (np.array): Origin (the P point) of the destination line of shape [2].
        q (np.array): Destination (the Q point) of the destination line of shape [2].
        x (np.array): The destination coords to calculate the uv for (the X point) of shape [2].

    Returns:
        (float, float):
            - The u coefficients for morphing for each coordinate.
            - The v coefficients for morphing for each coordinate.
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    u = ((x-p).dot(q-p)) / (norm(q-p)**2)
    v = ((x - p).dot(perp(q-p))) / (norm(q-p))
    return (u, v)
    #################################
    ######### DO NOT MODIFY #########
    #################################

def calculate_x_prime(p_prime, q_prime, u, v):
    """Find the source coordinates (X') from the source line (P', Q') and the u, v coefficients.

    This function should implement Equation 3 on page 36 of the Beier-Neely algorithm. 

    Args:
        p_prime (np.array): Origin (the P' point) of the source line of shape [2].
        q_prime (np.array): Destination (the Q' point) of the destination line of shape [2].
        u (float): The u coefficients for morphing.
        v (float): The v coefficients for morphing.

    Returns:
        (np.array): The source coordinates (X') of shape [2]. 
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    return p_prime + u.dot(q_prime - p_prime) + (v.dot(perp(q_prime - p_prime))) / (norm(q_prime - p_prime))
    #################################
    ######### DO NOT MODIFY #########
    #################################

def single_line_pair_algorithm(x, p, q, p_prime, q_prime):
    """Transform the destination coordinates (X) to the source (X') using the single line pair algorithm.

    This should implement the first pseudo-code from the top left of page 37 of the Beier-Neely paper.

    Args:
        x (np.array): The destination coordinates (X) of shape [2].
        p (np.array): Origin (the P point) of the destination line of shape [2].
        q (np.array): Destination (the Q point) of the destination line of shape [2].
        p_prime (np.array): Origin (the P' point) of the source line of shape [2].
        q_prime (np.array): Destination (the Q' point) of the source line of shape [2].
    
    Returns:
        (np.array): The source coordinates (X') of shape [2]
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    u, v = calculate_uv(p, q, x)
    return calculate_x_prime(p_prime, q_prime, u, v)
    #################################
    ######### DO NOT MODIFY #########
    #################################
    

def multiple_line_pair_algorithm(x, ps, qs, ps_prime, qs_prime, param_a, param_b, param_p):
    """Transform the destination coordinates (X) to the source (X') using the multiple line pair algorithm.

    This function should implement the pseudo code on the bottom right of page 37 of the Beier-Neely paper.

    Args: 
        x (np.array): The destination coordinates (X) of shape [2].
        ps (np.array): Origin (the P point) of the destination line of shape [num_lines, 2].
        qs (np.array): Destination (the Q point) of the destination line of shape [num_lines, 2].
        ps_prime (np.array): Origin (the P' point) of the source line of shape [num_lines, 2].
        qs_prime (np.array): Destination (the Q' point) of the source line of shape [num_lines, 2].
        param_a (float): The `a` parameter from the Beier-Neely paper controlling morph strength.
        param_b (float): The `b` parameter from the Beier-Neely paper controlling relative
                         line morph strength by distance.
        param_p (float): The `p` parameter from the Beier-Neely paper controlling relative
                         line morph strength by line length.
    
    Returns:
        (np.array): The source coordinates (X') of shape [2]
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    dsum, wsum = np.zeros(2), 0
    n_lines = ps.shape[0]
    for i in range(n_lines):
        pi, qi = ps[i], qs[i]
        u, v = calculate_uv(pi, qi, x)
        x_prime_i = calculate_x_prime(p_prime, q_prime, u, v)
        d_i = x_prime_i - x
        dist = norm(v)  # TODO: based on image, need to verify 
        length = norm(qi - pi)
        weight = ((length ** param_p) / (param_a + dist)) ** param_t
        dsum += d_i * weight
        wsum += weight
    x_prime = x + (dsum / wsum)
    return x_prime

    #################################
    ######### DO NOT MODIFY #########
    #################################

def linear(x, delta=1):
    """
    Helper function for bilinear interpolation. Returns the 1D linear 
    interpolation kernel

    Args:
        x (np.array): The source coordinates (X) of shape [2] in [-1, 1] coordinates.

    Returns: 
        float: The linear interpolation filter.
    """
    # TODO: delta as 1 for now, could change later?
    l2 = norm(x)
    if l2 >= 0 and l2 < delta:
        return 1 - l2 / delta
    return 0


def interpolate_at_x(source_image, x, bilinear=False):
    """Interpolates the source_image at some location x.
    
    Args:
        source_image (np.array): The source image of shape [H, W, 4]
        x (np.array): The source coordinates (X) of shape [2] in [-1, 1] coordinates.
        bilinear (bool): If true, will turn on bilinear sampling.
    
    Returns:
        (np.array): The source pixel of shape [4].
    """
    h, w = source_image.shape[:2]
    
    # [0, w] and [0, h] in floats
    pixel_float = img_ops.unnormalize_coordinates(x, h, w)
    import pdb; pdb.set_trace()
    
    if bilinear:
        ################################
        ####### PUT YOUR CODE HERE #####
        ################################
        pixel_int = np.convolve(pixel_float, h(x[0]) * h(x[1]))
        c, r = list(np.round(pixel_int))
        if c >= 0 and r >= 0 and c < w and r < h:
            return source_image[r, c]
        else:
            return np.zeros([4])
        #################################
        ######### DO NOT MODIFY #########
        #################################
    else:
        # Nearest neighbour interpolation
        # [0, w] and [0, h] in integers
        # We round, because the X.0 boundaries are the pixel centers. We select the nearest pixel centers.
        # When you implement bilinear interpolation, make sure you handle this correctly... samples can on
        # either sides of the pixel center!
        pixel_int = np.round(pixel_float).astype(np.int32)
        c, r = list(pixel_int)
        if c >= 0 and r >= 0 and c < w and r < h:
            return source_image[r, c]
        else:
            return np.zeros([4])

def backward_mapping(source_image, destination_image, source_morph_lines, destination_morph_lines,
                     param_a, param_b, param_p, supersampling, bilinear, vectorize):
    """Perform backward mapping onto the destination image.
    
    Args:
        source_image (np.ndarray): The source image of shape [H, W, 4]
        destination_image (np.ndarray): The destination image of shape [H, W, 4]
        source_morph_lines (np.ndarray): [N, 2, 2] tensor of coordinates for lines. The format is:
                                         [num_lines, (origin, destination), (x, y)]
        destination_morph_lines (np.ndarray): [N, 2, 2] tensor of coordinates for lines. The format is:
                                              [num_lines, (origin, destination), (x, y)]
        param_a (float): The `a` parameter from the Beier-Neely paper controlling morph strength.
        param_b (float): The `b` parameter from the Beier-Neely paper controlling relative
                         line morph strength by distance.
        param_p (float): The `p` parameter from the Beier-Neely paper controlling relative
                         line morph strength by line length.
        supersampling (int): The patch size for supersampling. 
        bilinear (bool): If True, will use bilinear interpolation on the pixels.
        vectorize (bool): If True, will use the vectorized version of the code (optional).

     Returns:
         (np.ndarray): [H, W, 4] image with the source image projected onto the destination image.
    """
    h, w, _ = destination_image.shape
    assert(source_image.shape[0] == h)
    assert(source_image.shape[1] == w)

    # The h, w, 4 buffer to populate and return
    output_buffer = np.zeros_like(destination_image)
    
    # The integer coordinates which you can access via xs_int[r, c]
    xs_int = img_ops.create_coordinates(h, w)

    # The float coordinates [-1, 1] which you can access via xs[r, c]
    # To avoid confusion, you should always denormalize this using img_ops.denormalize_coordinates(xs, h, w)
    # which will bring it back to pixel space, and avoid doing any pixel related operations (like filtering,
    # interpolation, etc) in normalized space. Normalized space however is nice for heavy numerical operations
    # for floating point precision reasons.
    xs = img_ops.normalize_coordinates(xs_int, h, w) 

    # Unpack the line tensors into the start and end points of the line
    ps = destination_morph_lines[:, 0]
    qs = destination_morph_lines[:, 1]
    ps_prime = source_morph_lines[:, 0]
    qs_prime = source_morph_lines[:, 1]
    
    if not vectorize:
        print("Algorithm running without vectorization...")
        print("")
        for r in range(h):
            # tqdm (a progress bar library) doesn't work great with kivy, 
            # so we implment our own progress bar here.
            # you should ignore this code for the most part.
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            percent_done = float(r)/float(h-1)
            print(f"[{'#' * int(percent_done*30)}{'-' * (30-int(percent_done*30))}] {int(100*percent_done)}% done")

            for c in range(w):
                x = xs[r, c][None]

                if supersampling > 1:
                    ################################
                    ####### PUT YOUR CODE HERE #####
                    ################################
                    pass
                    #################################
                    ######## DO NOT MODIFY #########
                    #################################
                else:
                    ################################
                    ####### PUT YOUR CODE HERE #####
                    ################################
                    pass
                    #################################
                    ######### DO NOT MODIFY #########
                    #################################

    else:
        ###########################################
        ### OPTIONAL OPTIONAL OPTIONAL OPTIONAL ###
        ### OPTIONAL VECTORIZED IMPLEMENTATION  ###
        ### OPTIONAL OPTIONAL OPTIONAL OPTIONAL ###
        ###########################################
        pass
        #################################
        ######### DO NOT MODIFY #########
        #################################

    return output_buffer

