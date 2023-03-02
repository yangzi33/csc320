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
import importlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import cv2
import scipy
import tqdm
import pandas as pd
import viscomp
import viscomp.ops.image as img_ops
import viscomp.ops.csv as csv_ops
import base_app

class Assignment2App(base_app.BaseAssignmentApp):
    """This is the app implemented for assignment 2. 

    You should not have to touch this for the most part, but reading through this code would
    likely be useful for debugging purposes as you implement the assignment.
    """
    def draw_image(self):
        """Draws the image onto the canvas.
        """
        self.source_image = np.flip(img_ops.read_image(self.args.source_image_path), 0)

        if not self.args.destination_image_path:
            self.destination_image = np.zeros_like(self.source_image)
        else:
            self.destination_image = np.flip(img_ops.read_image(self.args.destination_image_path), 0)
            if self.destination_image.shape[0] != self.source_image.shape[0] or \
               self.destination_image.shape[1] != self.destination_image.shape[1]:
                print("The left image dimensions should match the right image dimensions."
                     f" Got {self.source_image.shape} and {self.destination_image.shape}.")
                sys.exit(0)
        self.buffer.fill(0)
        h, w = self.app_resolution 
        screen_ratio = h/(w//2)
        input_ratio = self.source_image.shape[0] / self.source_image.shape[1]
        
        if self.debug_grid:
            hi, wi = self.source_image.shape[:2]
            for i in range(10):
                self.source_image[i*(hi//10), :] = 0
                self.source_image[:, i*(wi//10)] = 0
            hi, wi = self.destination_image.shape[:2]
            for i in range(10):
                self.destination_image[i*(hi//10), :] = 0
                self.destination_image[:, i*(wi//10)] = 0

        
        if input_ratio < screen_ratio:
            # resize to the height
            self.rescale_ratio = self.source_image.shape[1] / (w/2.0)
            self.rescaled_height = int(self.source_image.shape[0] * (1.0/self.rescale_ratio))
            self.rescaled_width = w//2
            offset = (h - self.rescaled_height) // 2
            self.y_min = offset
            self.y_max = offset+self.rescaled_height
            self.x_min = 0
            self.x_max = (w//2)
        else:
            self.rescale_ratio = self.source_image.shape[0] / h
            self.rescaled_height = h
            self.rescaled_width = int(self.source_image.shape[1] * (1.0/self.rescale_ratio))
            offset = ((w//2) - self.rescaled_width) // 2
            self.y_min = 0
            self.y_max = h
            self.x_min = offset
            self.x_max = offset+self.rescaled_width
        
        # S
        self.rescaled_source = cv2.resize(self.source_image, (self.rescaled_width, self.rescaled_height), interpolation=cv2.INTER_AREA)
        # D
        self.rescaled_destination = cv2.resize(self.destination_image, (self.rescaled_width, self.rescaled_height), interpolation=cv2.INTER_AREA)
        # M
        self.warped_source = np.zeros_like(self.source_image)
        self.warped_destination = np.zeros_like(self.source_image)
        self.rescaled_warped_destination = np.zeros_like(self.rescaled_source)
        self.rescaled_warped_source = np.zeros_like(self.rescaled_source)
        self.morphed_image = np.zeros_like(self.source_image)
        self.rescaled_morphed_image = np.zeros_like(self.rescaled_source)

        self.update_buffer()


    def update_buffer(self):
        """Updates the buffer.
        """
        h, w = self.app_resolution 
        if self.left_buffer == 'S':
            self.buffer[self.y_min:self.y_max, self.x_min:self.x_max] = self.rescaled_source
        elif self.left_buffer == 'WD':
            self.buffer[self.y_min:self.y_max, self.x_min:self.x_max] = self.rescaled_warped_destination
        elif self.left_buffer == 'M':
            self.buffer[self.y_min:self.y_max, self.x_min:self.x_max] = self.rescaled_morphed_image

        if self.right_buffer == 'D':
            self.buffer[self.y_min:self.y_max, (w//2)+self.x_min:(w//2)+self.x_max] = self.rescaled_destination
        elif self.right_buffer == 'WS':
            self.buffer[self.y_min:self.y_max, (w//2)+self.x_min:(w//2)+self.x_max] = self.rescaled_warped_source
        elif self.right_buffer == 'M':
            self.buffer[self.y_min:self.y_max, (w//2)+self.x_min:(w//2)+self.x_max] = self.rescaled_morphed_image

    def check_image_bounds(self, coord):
        """Checks if a coordinate is within the bounds of the image on screen.
        
        Args:
            coord (np.array): 2D coordinate of shape [2] in integer coordinates.

        Returns:
            (bool, str)
            - True if within bounds
            - The side of the image that the coord is on, "left" or "right"
        """
        h, w = self.app_resolution
        if coord[0] < self.x_max and coord[0] > self.x_min and \
           coord[1] < self.y_max and coord[1] > self.y_min:
            return True, "left"
        elif coord[0] < self.x_max + (w//2) and coord[0] > self.x_min + (w//2) and \
           coord[1] < self.y_max and coord[1] > self.y_min:
            return True, "right"
        else:
            if coord[0] < self.x_max and coord[0] > self.x_min:
                return False, "left"
            elif coord[0] < self.x_max + (w//2) and coord[0] > self.x_min + (w//2):
                return False, "right"
            else:
                return False, None

    def canvas_to_image(self, coords, side):
        """Remaps the canvas-space coordinates to the image-space coordinates.
        
        Args:
            coords (np.array): 2D coordinate of shape [N, 2] in integer coordinates.

        Returns:
            (np.array): 2D coordinate of shape [N, 2] in integer coordinates.
        """
        h, w = self.app_resolution
        if side == "left":
            image_coords = np.concatenate([
                coords[..., 0:1] - self.x_min,
                coords[..., 1:2] - self.y_min], axis=-1)
            image_coords = image_coords.astype(np.float32) * self.rescale_ratio
            return image_coords.astype(np.int32)
        elif side == "right":
            image_coords = np.concatenate([
                coords[..., 0:1] - self.x_min - (w//2),
                coords[..., 1:2] - self.y_min], axis=-1)
            image_coords = image_coords.astype(np.float32) * self.rescale_ratio
            return image_coords.astype(np.int32)
        else:
            return np.zeros_like(coords)
    
    def image_to_canvas(self, coords, side):
        """Remaps the image-space coordinates to the canvas-space coordinates.
        
        Args:
            coords (np.array): 2D coordinate of shape [N, 2] in integer coordinates.

        Returns:
            (np.array): 2D coordinate of shape [N, 2] in integer coordinates.
        """
        h, w = self.app_resolution
        if side == "left":
            canvas_coords = coords.astype(np.float32) / self.rescale_ratio
            canvas_coords = np.concatenate([
                canvas_coords[..., 0:1] + self.x_min,
                canvas_coords[..., 1:2] + self.y_min], axis=-1)
            return canvas_coords.astype(np.int32)
        elif side == "right":
            canvas_coords = coords.astype(np.float32) / self.rescale_ratio
            canvas_coords = np.concatenate([
                canvas_coords[..., 0:1] + self.x_min + (w//2),
                canvas_coords[..., 1:2] + self.y_min], axis=-1)
            return canvas_coords.astype(np.int32)
        else:
            return np.zeros_like(coords)


    def init_state(self, first=True):
        """Initialize a bunch of random variables.
        """
        self.left_buffer = 'S'
        if self.args.destination_image_path:
            self.right_buffer = 'D'
        else:
            self.right_buffer = 'WS'
        self.print_help_text()
        self.dirty_canvas = False
        
        self.raster_line_buffers = []
        self.raster_point_buffers = []
        self.source_morph_lines = np.zeros([0, 2, 2])
        self.destination_morph_lines = np.zeros([0, 2, 2])
        self.source_morph_index = 0
        self.destination_morph_index = 0

        self.mouse_location = np.array([0,0])
        self.mouse_location_int = np.array([0,0])
        self.mouse_image_location = np.array([0,0])
        self.mouse_image_location_int = np.array([0,0])
        self.debug_grid = self.args.debug_grid
        self.draw_image()

        self.param_a = self.args.param_a
        self.param_b = self.args.param_b
        self.param_p = self.args.param_p
        self.param_t = self.args.param_t
        self.num_frames = self.args.num_frames

        self.show_source_morph_lines = True
        self.show_destination_morph_lines = True

        self.select_mode = False
        self.selected_idx = 0
        self.supersampling = self.args.supersampling
        self.vectorize = self.args.vectorize
        self.run_realtime = self.args.run_realtime
        self.bilinear = self.args.bilinear
        self.edit_arrows = False
        self.verbose = self.args.verbose
        self.show_warped_cursor = self.args.show_warped_cursor
        self.invalid_config = True

        self.focus_pane = "none"

        if first and self.args.source_line_path:
            self.source_morph_lines = csv_ops.load_csv(self.args.source_line_path).reshape(-1, 2, 2)
            self.source_morph_lines = img_ops.normalize_coordinates(
                    self.image_to_canvas(self.source_morph_lines, "left"), *self.app_resolution)
            self.source_morph_index = self.source_morph_lines.shape[0] * self.source_morph_lines.shape[1]
            
            if not self.args.destination_line_path:
                self.destination_morph_lines = np.copy(self.source_morph_lines)
                self.destination_morph_lines[..., 0] += 1.0
                self.destination_morph_index = self.source_morph_index

            else:
                self.destination_morph_lines = csv_ops.load_csv(self.args.destination_line_path).reshape(-1, 2, 2)
                self.destination_morph_lines = img_ops.normalize_coordinates(
                        self.image_to_canvas(self.destination_morph_lines, "right"), *self.app_resolution)
                self.destination_morph_index = self.destination_morph_lines.shape[0] * self.destination_morph_lines.shape[1]

    def clear_arrows(self):
        """Clear arrows"""
        self.raster_line_buffers = []
        self.raster_point_buffers = []
        self.source_morph_lines = np.zeros([0, 2, 2])
        self.destination_morph_lines = np.zeros([0, 2, 2])
        self.source_morph_index = 0
        self.destination_morph_index = 0


    def add_raster_arrow(self, origin, destination, color, line_width):
        """Small wrapper on top of add_raster_line that draws an arrow.
        """
        head_0, head_1 = private_helpers.get_arrow_heads(origin, destination)
        self.add_raster_line(origin, destination, color, line_width)
        self.add_raster_line(destination, head_0, color, line_width)
        self.add_raster_line(destination, head_1, color, line_width)

    def add_morph_lines(self, morph_lines, morph_index, palette, right):
        """Helper function to draw the morph lines.

        Args:
            morph_lines (np.array): Tensor to hold lines of shape [N, 2, 2], where each element is a 
                                    [x, y] point. The format is: [num_lines, (origin, destination), (x, y)].
            morph_index (int): The number of clicks made so far. 
            palette (np.array): [N, 3] tensor of colors to use to color the lines.
            right (bool): Bool indicating the side of the image.
        """
        num_full_lines = morph_index//2
        num_colors = palette.shape[0] 
        for i in range(num_full_lines):
            self.add_raster_arrow(morph_lines[i, 0], morph_lines[i, 1], palette[i%num_colors], line_width=4)
        if morph_index % 2 == 1:
            if right and self.focus_pane == "left":
                self.add_raster_arrow(morph_lines[-1, 0], np.array([self.mouse_location[0]+1.0, self.mouse_location[1]]), 
                                      palette[num_full_lines%num_colors], line_width=4)
            elif right and self.focus_pane == "right":
                self.add_raster_arrow(morph_lines[-1, 0], np.array([self.mouse_location[0], self.mouse_location[1]]), 
                                      palette[num_full_lines%num_colors], line_width=4)
            elif not right and self.focus_pane == "left":
                self.add_raster_arrow(morph_lines[-1, 0], self.mouse_location, 
                                      palette[num_full_lines%num_colors], line_width=4)
            elif not right and self.focus_pane == "right":
                self.add_raster_arrow(morph_lines[-1, 0], np.array([self.mouse_location[0]-1.0, self.mouse_location[1]]),
                                      palette[num_full_lines%num_colors], line_width=4)

    def render(self):
        """Main render loop.

        This function will run the algorithm by calling viscomp.algos.run_a2_algos. The results will then 
        be drawn to self.buffer, which Glumpy takes and draws to the screen. There are also some extra
        raster graphics drawn on the screen.
        """
        palette = np.array([
            [246, 51, 162, 255],
            [248, 146, 33, 255],
            [40, 214, 192, 255],
            [249, 226, 44, 255]]) / 255.0
        num_colors = palette.shape[0]

        h, w, _ = self.buffer.shape
        num_lines = min(self.source_morph_index // 2, self.destination_morph_index // 2)
        within_bound, side = self.check_image_bounds(self.mouse_location_int)

        if self.dirty_canvas or self.run_realtime:
            if self.verbose:
                print("Running the morphing algorithm...")
            if num_lines > 0:
                source_morph_lines = img_ops.unnormalize_coordinates(self.source_morph_lines[:num_lines], h, w)
                destination_morph_lines = img_ops.unnormalize_coordinates(self.destination_morph_lines[:num_lines], h, w)
                source_morph_lines = img_ops.normalize_coordinates(
                        self.canvas_to_image(source_morph_lines, "left"), *self.source_image.shape[:2])
                destination_morph_lines = img_ops.normalize_coordinates(
                        self.canvas_to_image(destination_morph_lines, "right"), *self.source_image.shape[:2])
                
                self.warped_source = algos.run_a2_algo(self.source_image, self.destination_image, 
                                                 source_morph_lines, destination_morph_lines,
                                                 self.param_a, self.param_b, self.param_p,
                                                 self.supersampling, self.bilinear, 
                                                 False, self.param_t, self.vectorize)
                
                if self.args.destination_image_path:
                    self.warped_destination = algos.run_a2_algo(self.destination_image, self.source_image, 
                                                     destination_morph_lines, source_morph_lines,
                                                     self.param_a, self.param_b, self.param_p,
                                                     self.supersampling, self.bilinear, 
                                                     False, 1.0 - self.param_t, self.vectorize)
                    self.warped_destination[..., -1] = 1.0
                    
                    self.morphed_image = self.warped_source * (1.0 - self.param_t) + \
                            self.warped_destination * self.param_t
                    self.morphed_image[..., -1] = 1.0

                    self.rescaled_warped_destination = cv2.resize(self.warped_destination, 
                            (self.rescaled_width, self.rescaled_height), interpolation=cv2.INTER_AREA)
                    
                    self.rescaled_morphed_image = cv2.resize(self.morphed_image, 
                            (self.rescaled_width, self.rescaled_height), interpolation=cv2.INTER_AREA)

                self.warped_source[..., -1] = 1.0
                self.rescaled_warped_source = cv2.resize(self.warped_source, 
                        (self.rescaled_width, self.rescaled_height), interpolation=cv2.INTER_AREA)
                self.update_buffer()
                if self.verbose:
                        print("Finished the morphing algorithm!")
                self.dirty_canvas = False

        if num_lines > 0 and self.show_warped_cursor and within_bound:
            source_morph_lines = img_ops.unnormalize_coordinates(self.source_morph_lines[:num_lines], h, w)
            destination_morph_lines = img_ops.unnormalize_coordinates(self.destination_morph_lines[:num_lines], h, w)
            source_morph_lines = img_ops.normalize_coordinates(
                    self.canvas_to_image(source_morph_lines, "left"), *self.source_image.shape[:2])
            destination_morph_lines = img_ops.normalize_coordinates(
                    self.canvas_to_image(destination_morph_lines, "right"), *self.source_image.shape[:2])
            ps = destination_morph_lines[:, 0]
            qs = destination_morph_lines[:, 1]
            ps_prime = source_morph_lines[:, 0]
            qs_prime = source_morph_lines[:, 1]
            invalid_config = False
            if self.left_buffer == "S" and self.right_buffer == "D":
                invalid_config = True
            elif self.left_buffer == "WD" and self.right_buffer == "D":
                invalid_config = True
            elif self.left_buffer == "WD" and self.right_buffer == "WS":
                invalid_config = True
            elif self.right_buffer == "M":
                invalid_config = True
            elif self.left_buffer == "S" and self.right_buffer == "WS":
                if side == "left":
                    mouse_project = algos.multiple_line_pair_algorithm(self.mouse_image_location, 
                                               ps_prime, qs_prime, ps, qs,
                                               self.param_a, self.param_b, self.param_p)
                if side == "right":
                    mouse_project = algos.multiple_line_pair_algorithm(self.mouse_image_location, 
                                                       ps, qs, ps_prime, qs_prime,
                                                       self.param_a, self.param_b, self.param_p)
            elif self.left_buffer == "WD" and self.right_buffer == "D":
                if side == "left":
                    mouse_project = algos.multiple_line_pair_algorithm(self.mouse_image_location, 
                                               ps, qs, ps_prime, qs_prime,
                                               self.param_a, self.param_b, self.param_p)
                if side == "right":
                    mouse_project = algos.multiple_line_pair_algorithm(self.mouse_image_location, 
                                               ps_prime, qs_prime, ps, qs,
                                               self.param_a, self.param_b, self.param_p)

            if not invalid_config:
                self.invalid_config = invalid_config
                self.mouse_project = mouse_project
                mouse_project = img_ops.unnormalize_coordinates(mouse_project[None], *self.source_image.shape[:2])
                self.mouse_project_int = mouse_project[0].astype(np.int32)
                mouse_project = self.image_to_canvas(mouse_project, side)
                mouse_project = img_ops.normalize_coordinates(mouse_project, h, w)[0]
                if side == "left":
                    mouse_project[0] += 1.0
                if side == "right":
                    mouse_project[0] -= 1.0
                
                if not invalid_config:
                    self.add_raster_point(mouse_project, 
                                          palette[(self.source_morph_index//2) % num_colors], point_size=4)

        self.add_raster_line([0,-1], [0, 1], [1.0, 1.0, 1.0, 1.0], line_width=4)
        if within_bound and side == "left":
            self.add_raster_point(self.mouse_location, 
                                  palette[(self.source_morph_index//2) % num_colors], point_size=4)
        elif within_bound and side == "right":
            self.add_raster_point(self.mouse_location, 
                                  palette[(self.destination_morph_index//2) % num_colors], point_size=4)
        
        if self.show_source_morph_lines:
            self.add_morph_lines(self.source_morph_lines, self.source_morph_index, palette, False)
        if self.show_destination_morph_lines:
            self.add_morph_lines(self.destination_morph_lines, self.destination_morph_index, palette, True)
        

    def on_key_press(self, symbol, modifiers):
        """Callback for key press. Unused in this assignment.
        """
        #print('Key pressed (symbol=%s, modifiers=%s)' % (symbol, modifiers))
        if symbol == 67: #c
            self.clear_arrows()
        elif symbol == 82: #r
            self.dirty_canvas = True
        elif symbol == 72: #h
            self.print_help_text()
        elif symbol == 69: # e
            self.export()

    def export(self):
        """Function to export output image and csv files.
        """
        if not self.args.output_path:
            print("Export failed. Please set the argument --output-path")
        else:
            h, w = self.app_resolution
            num_lines = min(self.source_morph_index // 2, self.destination_morph_index // 2)
            if num_lines > 0:
                source_image_name = os.path.splitext(os.path.basename(self.args.source_image_path))[0]
                if self.args.destination_image_path:
                    destination_image_name = os.path.splitext(os.path.basename(self.args.destination_image_path))[0]
                    source_image_name += "_" + destination_image_name
                source_morph_lines = img_ops.unnormalize_coordinates(self.source_morph_lines[:num_lines], h, w)
                destination_morph_lines = img_ops.unnormalize_coordinates(self.destination_morph_lines[:num_lines], h, w)
                source_morph_lines = self.canvas_to_image(source_morph_lines, "left").astype(np.int32)
                destination_morph_lines = self.canvas_to_image(destination_morph_lines, "right").astype(np.int32)
                csv_ops.write_csv(source_morph_lines.reshape(-1, 4), os.path.join(self.args.output_path, f'{source_image_name}_source.csv'))
                csv_ops.write_csv(destination_morph_lines.reshape(-1, 4), os.path.join(self.args.output_path, f'{source_image_name}_destination.csv'))
                img_ops.write_image(np.flip(self.warped_source, 0), os.path.join(self.args.output_path, f'{source_image_name}_warpedsource.png'))

                if self.args.destination_image_path:
                    img_ops.write_image(np.flip(self.warped_destination, 0), os.path.join(self.args.output_path, f'{source_image_name}_warpeddestination.png'))
                    img_ops.write_image(np.flip(self.morphed_image, 0), os.path.join(self.args.output_path, f'{source_image_name}_morphed.png'))

                print(f"Exported image and csv files to {os.path.abspath(self.args.output_path)}")
            else:
                print("Export failed. No lines drawn on the canvas.")

    def export_video(self):
        """Function to export a video of the interpolation.
        """
        if not self.args.output_path:
            print("Export failed. Please set the argument --output-path")
        else:
            h, w = self.app_resolution[:2]
            num_lines = min(self.source_morph_index // 2, self.destination_morph_index // 2)
            
            if num_lines > 0:
                source_morph_lines = img_ops.unnormalize_coordinates(self.source_morph_lines[:num_lines], h, w)
                destination_morph_lines = img_ops.unnormalize_coordinates(self.destination_morph_lines[:num_lines], h, w)
                source_morph_lines = img_ops.normalize_coordinates(
                        self.canvas_to_image(source_morph_lines, "left"), *self.source_image.shape[:2])
                destination_morph_lines = img_ops.normalize_coordinates(
                        self.canvas_to_image(destination_morph_lines, "right"), *self.source_image.shape[:2])
            
                frame_ts = np.arange(0, self.num_frames+1).astype(np.float32) / (self.num_frames)
                
                source_image_name = os.path.splitext(os.path.basename(self.args.source_image_path))[0]
                if self.args.destination_image_path:
                    destination_image_name = os.path.splitext(os.path.basename(self.args.destination_image_path))[0]
                    source_image_name += "_" + destination_image_name

                
                print("Generating frames... (this will usually take a while)")
                print("")
                for i, t in enumerate(list(frame_ts)):
                    sys.stdout.write('\x1b[1A')
                    sys.stdout.write('\x1b[2K')
                    percent_done = float(i)/float(frame_ts.shape[0]-1)
                    print(f"[{'#' * int(percent_done*30)}{'-' * (30-int(percent_done*30))}] {int(100*percent_done)}% done")
                    warped_source = algos.run_a2_algo(self.source_image, self.destination_image, 
                                                     source_morph_lines, destination_morph_lines,
                                                     self.param_a, self.param_b, self.param_p,
                                                     self.supersampling, self.bilinear, 
                                                     False, t, self.vectorize)
                    warped_source[..., -1] = 1.0
                    img_ops.write_image(np.flip(warped_source, 0), os.path.join(self.args.output_path, f'{source_image_name}_warpedsource_{i:06d}.png'))
                    
                    if self.args.destination_image_path:
                        warped_destination = algos.run_a2_algo(self.destination_image, self.source_image, 
                                                         destination_morph_lines, source_morph_lines,
                                                         self.param_a, self.param_b, self.param_p,
                                                         self.supersampling, self.bilinear, 
                                                         False, 1.0 - t, self.vectorize)
                        
                        warped_destination[..., -1] = 1.0
                        morphed_image = warped_source * (1.0 - t) + warped_destination * t
                        morphed_image[..., -1] = 1.0
                        img_ops.write_image(np.flip(warped_destination, 0), os.path.join(self.args.output_path, f'{source_image_name}_warpeddestination_{i:06d}.png'))
                        img_ops.write_image(np.flip(morphed_image, 0), os.path.join(self.args.output_path, f'{source_image_name}_morphed_{i:06d}.png'))
                        
                print(f"Algorithm success. Wrote frames to {self.args.output_path}.")

    def on_key_release(self, symbol, modifiers):
        """Callback for key release. Unused in this assignment.
        """
        pass
    
    def print_help_text(self):
        """Prints the help text.
        """
        print("CONTROLS:")
        print("   c : Clears the arrows drawn on the screen.")
        print("   h : Shows this help text.")
        print("   r : Runs the morphing algorithm.")

    def get_current_state_text(self):
        """Get the current state text as a list of lines.

        Returns:
            (list of str): The state text.
        """
        texts = ["aaa"]
        return texts

    def print_current_state(self):
        """Print out the current state out onto the CLI.
        """
        print("")
        text = self.get_current_state_text()
        for t in text:
            print(t)
    
    def on_mouse_release(self, x, y, button):
        """Call back for mouse release.
        """
        if self.select_mode:
            self.select_mode = False
            #self.dirty_canvas = True
            self.focus_pane = "none"
    
    def on_mouse_drag(self, x, y, dx, dy, button):
        """Callback for mouse drag.
        """
        self.update_mouse_location(x, y)
        if self.select_mode:
            if self.mouse_within_bound and self.mouse_side == "right" and self.focus_pane == "right":
                self.destination_morph_lines[self.selected_idx//2, self.selected_idx%2] = np.array(self.mouse_location)
            elif self.mouse_within_bound and self.mouse_side == "left" and self.focus_pane == "left":
                self.source_morph_lines[self.selected_idx//2, self.selected_idx%2] = np.array(self.mouse_location)

    def on_mouse_motion(self, x, y, dx, dy):
        """Callback for mouse motion.
        """
        self.update_mouse_location(x, y)

    def update_mouse_location(self, x, y):
        """Update mouse location.
        """
        scaled_x = 2.0 * (x / self.app_resolution[1]) - 1.0
        scaled_y = 2.0 * (y / self.app_resolution[0]) - 1.0
        scaled_y *= -1
        self.mouse_location_int = [int(x), int(self.app_resolution[0] - y)]
        self.mouse_location = [scaled_x, scaled_y]
        self.mouse_within_bound, self.mouse_side = self.check_image_bounds(self.mouse_location_int)
        if self.mouse_within_bound:
            self.mouse_image_location_int = self.canvas_to_image(np.array([self.mouse_location_int]), self.mouse_side)[0]
            self.mouse_image_location = img_ops.normalize_coordinates(self.mouse_image_location_int, *self.source_image.shape[:2])

    def on_mouse_press(self, x, y, button):
        """Callback for mouse click.
        """
        self.update_mouse_location(x, y)
        if self.verbose:
            print("Mouse was clicked!")
            print(f"    Side: {self.mouse_side}")
            print(f"    Click location (on canvas)")
            print(f"        int   : {self.mouse_location_int}")
            print(f"        float : [{self.mouse_location[0]:.2f}, {self.mouse_location[1]:.2f}]")
            if self.mouse_within_bound:
                print(f"    Click location (on image)")
                print(f"        int   : {self.mouse_image_location_int}")
                print(f"        float : [{self.mouse_image_location[0]:.2f}, {self.mouse_image_location[1]:.2f}]")
                if not self.invalid_config and self.show_warped_cursor:
                    print(f"    Click location (on warped image)")
                    print(f"        int   : {self.mouse_project_int}")
                    print(f"        float : [{self.mouse_project[0]:.2f}, {self.mouse_project[1]:.2f}]")

        
        if not self.edit_arrows:
            if self.mouse_within_bound and self.mouse_side == "left" and self.focus_pane != "right":
                if self.source_morph_index % 2 == 0:
                    self.source_morph_lines = \
                            np.concatenate([self.source_morph_lines, 
                                            np.array([[self.mouse_location, [0, 0]]])], axis=0)
                    self.destination_morph_lines = \
                            np.concatenate([self.destination_morph_lines,
                                            np.array([[[self.mouse_location[0] + 1.0, self.mouse_location[1]],
                                                       [0, 0]]])], axis=0)
                    self.focus_pane = "left"
                else:
                    self.source_morph_lines[-1, 1] = np.array(self.mouse_location)
                    self.destination_morph_lines[-1, 1] = np.array([self.mouse_location[0] + 1.0, self.mouse_location[1]])
                    self.focus_pane = "none"
                    
                self.source_morph_index += 1
                self.destination_morph_index += 1
            
            if self.mouse_within_bound and self.mouse_side == "right" and self.focus_pane != "left":
                if self.destination_morph_index % 2 == 0:
                    self.destination_morph_lines = \
                            np.concatenate([self.destination_morph_lines, 
                                            np.array([[self.mouse_location, [0, 0]]])], axis=0)
                    self.source_morph_lines = \
                            np.concatenate([self.source_morph_lines,
                                            np.array([[[self.mouse_location[0] - 1.0, self.mouse_location[1]],
                                                       [0, 0]]])], axis=0)
                    self.focus_pane = "right"
                else:
                    self.destination_morph_lines[-1, 1] = np.array(self.mouse_location)
                    self.source_morph_lines[-1, 1] = np.array([self.mouse_location[0] - 1.0, self.mouse_location[1]])
                    self.focus_pane = "none"
                    
                self.source_morph_index += 1
                self.destination_morph_index += 1
            
        else:
            if self.mouse_side == "left":
                if self.source_morph_index >= 2:
                    disps = (self.source_morph_lines - np.array(self.mouse_location)).reshape(-1, 2)
                    dists = (disps**2).sum(-1)
                    if np.min(dists) < 0.025:
                        self.selected_idx = np.argmin(dists)
                        self.select_mode = True
                        self.focus_pane = "left"
            else:
                if self.destination_morph_index >= 2:
                    disps = (self.destination_morph_lines - np.array(self.mouse_location)).reshape(-1, 2)
                    dists = (disps**2).sum(-1)
                    if np.min(dists) < 0.025:
                        self.selected_idx = np.argmin(dists)
                        self.select_mode = True
                        self.focus_pane = "right"


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
    app_group.add_argument('--run-realtime', action='store_true',
                           help='Set this to run the morphing every frame of the app at 30 FPS. Will only work if you have' 
                                ' an optimized implementation with a small image.')
    app_group.add_argument('--output-path', type=str, default="results",
                           help='Path to folder to export output image and csv files for the morph lines drawn on the images.')
    app_group.add_argument('--source-line-path', type=str,
                           help='Path to csv file to import preset left morph lines.') 
    app_group.add_argument('--destination-line-path', type=str,
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
    app_group.add_argument('--show-warped-cursor', action='store_true',
                           help='Set this to project the mouse pixel cursor onto the other image using the morph lines.') 
    app_group.add_argument('--num-frames', type=int, default=20,
                           help='The number of frames to output for interpolation.')
    args = parser.parse_args()
    import sourcedefender
    import viscomp.binaries.helpers as private_helpers
    if args.reference_solution:
        import viscomp.binaries as algos
    else:
        import viscomp.algos as algos

    from base_app import BaseAssignmentRunner
    window = Assignment2App(args)
    app = BaseAssignmentRunner(window)
    app.run()

