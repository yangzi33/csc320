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
import importlib

import numpy as np
from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt
from glumpy import app, gloo, gl, glm, data
import imgui
import cv2
import scipy

import viscomp
import viscomp.ops.image as img_ops

import base_app

class Assignment1App(base_app.BaseAssignmentApp):
    """This is the app implemented for assignment 1. 

    You should not have to touch this for the most part, but reading through this code would
    likely be useful for debugging purposes as you implement the assignment.
    """

    def init_state(self):
        """Initialize a bunch of random variables.
        """
        self.dirty_canvas = False
        self.right_pressed = False
        self.left_pressed = False
        self.forward_pressed = False
        self.backward_pressed = False
        self.raster_line_buffers = []
        self.raster_point_buffers = []
        self.left_corners = np.zeros([4, 2])
        self.left_corner_index = 0
        self.right_corners = np.zeros([4, 2])
        self.right_corner_index = 0
        self.backprojected_corners = np.zeros([4, 2])

        self.H_matrix = None
        self.mouse_location = np.array([0,0])
        self.mouse_location_int = np.array([0,0])
        self.input_image = img_ops.read_image(self.args.image_path)

        h, w = self.app_resolution 
        
        screen_ratio = h/(w//2)
        input_ratio = self.input_image.shape[0] / self.input_image.shape[1]
    
        # Draw the image to the screen
        if input_ratio < screen_ratio:
            # resize to the height
            ratio = self.input_image.shape[1] / (w/2.0)
            resized_height = int(self.input_image.shape[0] * (1.0/ratio))
            self.input_image = cv2.resize(self.input_image, ((w//2), resized_height), interpolation=cv2.INTER_AREA)
            offset = (h - self.input_image.shape[0]) // 2
            self.buffer[offset:offset+self.input_image.shape[0], 0:(w//2)] = np.flip(self.input_image, 0) 
        else:
            ratio = self.input_image.shape[0] / h
            resized_width = int(self.input_image.shape[1] * (1.0/ratio))
            self.input_image = cv2.resize(self.input_image, (resized_width, h), interpolation=cv2.INTER_AREA)
            offset = ((w//2) - self.input_image.shape[1]) // 2
            self.buffer[0:h, offset:offset+self.input_image.shape[1]] = np.flip(self.input_image, 0)

    def draw_imgui(self):
        """Draw the imgui menu.
        """
        imgui.new_frame()
        imgui.begin('assignment')

        text = "CSC320 Assignment 1"
        imgui.text(text)
        
        expanded, visible = imgui.collapsing_header("Debug", True, imgui.TREE_NODE_DEFAULT_OPEN)
        if expanded:
            
            imgui.text(f"Mouse Location: {self.mouse_location[0]:.02f}, {self.mouse_location[1]:.02f} "
                       f"({int(self.mouse_location_int[0])}, {int(self.mouse_location_int[1])})")

            state_texts = self.get_current_state_text()   
            for text in state_texts:
                imgui.text(text)
        
        imgui.set_window_size(450, 0)
        imgui.end()
        imgui.end_frame()
        imgui.render()

        self.clear()

    def add_corners(self, corners, corner_index, palette, right=True):
        """Helper function to draw the "corners" as raster graphics.
        """
        for i in range(corner_index):
            self.add_raster_point(corners[i], palette[i], point_size=4)
        
        for i in range(max(corner_index-1, 0)):
            self.add_raster_line(corners[i], corners[(i+1)%4], [0, 0, 1, 1], line_width=4)
        if corner_index == 4:
            self.add_raster_line(corners[3], corners[0], [0, 0, 1, 1], line_width=4)
        elif corner_index > 0:
            if right and self.mouse_location[0] > 0.0:
                self.add_raster_line(corners[corner_index-1], self.mouse_location, [0, 0, 1, 1], line_width=4)
            elif not right and self.mouse_location[0] < 0.0:
                self.add_raster_line(corners[corner_index-1], self.mouse_location, [0, 0, 1, 1], line_width=4)

    def render(self):
        """Main render loop.

        This function will run the algorithm by calling viscomp.algos.run_a1_algos. The results will then 
        be drawn to self.buffer, which Glumpy takes and draws to the screen. There are also some extra
        raster graphics drawn on the screen.
        """
        palette = np.array([
            [246, 51, 162, 255],
            [248, 146, 33, 255],
            [40, 214, 192, 255],
            [249, 226, 44, 255]]) / 255.0
        
        h, w, _ = self.buffer.shape

        if self.dirty_canvas:
            if self.left_corner_index == 4 and self.right_corner_index == 4:
                output = algos.run_a1_algo(self.buffer, self.buffer, self.left_corners, self.right_corners)
                self.buffer[:, w//2:] = output[:, w//2:]
                self.dirty_canvas = False
            else:
                self.buffer[:, w//2:] = 0.0
        
        self.add_raster_line([0,-1], [0, 1], [1.0, 1.0, 1.0, 1.0], line_width=4)
        self.add_raster_point(self.mouse_location, [1.0, 1.0, 1.0, 1.0], point_size=4)
        self.add_corners(self.left_corners, self.left_corner_index, palette, False)
        self.add_corners(self.right_corners, self.right_corner_index, palette, True)

    def on_key_press(self, symbol, modifiers):
        """Callback for key press. Unused in this assignment.
        """
        pass
        #print('Key pressed (symbol=%s, modifiers=%s)' % (symbol, modifiers))
    
    def on_key_release(self, symbol, modifiers):
        """Callback for key release. Unused in this assignment.
        """
        pass
    
    def get_current_state_text(self):
        """Get the current state text as a list of lines.

        Returns:
            (list of str): The state text.
        """
        texts = []
        colors = ["pink", "orng", "cyan", "yelw"]
        text = f"      Corners (l)    Corners (r)"
        if self.right_corner_index == 4 and self.left_corner_index == 4:
            text += f"    Backprojected (r)"
        texts.append(text)
        text =  f"       {'x'.rjust(5)} {'y'.rjust(5)}"
        text += f"    {'x'.rjust(5)} {'y'.rjust(5)}"
        if self.right_corner_index == 4 and self.left_corner_index == 4:
            text += f"    {'x'.rjust(5)} {'y'.rjust(5)}"
        texts.append(text)
        for i in range(4):
            if i < self.left_corner_index:
                x = int(((self.left_corners[i, 0] + 1.0) / 2.0) * self.screen['tex'].shape[1])
                y = int(((self.left_corners[i, 1] + 1.0) / 2.0) * self.screen['tex'].shape[0])
            else:
                x = "null"
                y = "null"
            text = f"{colors[i]}: [{str(x).rjust(5)} {str(y).rjust(5)}]"
            
            if i < self.right_corner_index:
                x = int(((self.right_corners[i, 0] + 1.0) / 2.0) * self.screen['tex'].shape[1])
                y = int(((self.right_corners[i, 1] + 1.0) / 2.0) * self.screen['tex'].shape[0])
            else:
                x = "null"
                y = "null"
            text += f"  [{str(x).rjust(5)} {str(y).rjust(5)}]"

            if self.right_corner_index == 4 and self.left_corner_index == 4:
                x = int(((self.backprojected_corners[i, 0] + 1.0) / 2.0) * self.screen['tex'].shape[1])
                y = int(((self.backprojected_corners[i, 1] + 1.0) / 2.0) * self.screen['tex'].shape[0])
                text += f"  [{str(x).rjust(5)} {str(y).rjust(5)}]"
            texts.append(text)
        return texts

    def print_current_state(self):
        """Print out the current state out onto the CLI.
        """
        print("")
        text = self.get_current_state_text()
        for t in text:
            print(t)

    def on_mouse_motion(self, x, y, dx, dy):
        """Callback for mouse motion.
        """
        scaled_x = 2.0 * (x / self.screen["tex"].shape[1]) - 1.0
        scaled_y = 2.0 * (y / self.screen["tex"].shape[0]) - 1.0
        self.mouse_location_int = [x, self.app_resolution[1] - y]
        self.mouse_location = [scaled_x, -scaled_y]

    def on_mouse_press(self, x, y, button):
        """Callback for mouse click.
        """
        scaled_x = 2.0 * (x / self.screen["tex"].shape[1]) - 1.0
        scaled_y = 2.0 * (y / self.screen["tex"].shape[0]) - 1.0
        scaled_y *= -1

        # Not exactly the most maintainable code. But is more clear, hopefully!
        if scaled_x < 0.0:
            if self.left_corner_index < 4:
                self.left_corners[self.left_corner_index, 0] = scaled_x 
                self.left_corners[self.left_corner_index, 1] = scaled_y 
                self.left_corner_index += 1
            else:
                self.left_corner_index = 0
                self.left_corners[self.left_corner_index, 0] = scaled_x 
                self.left_corners[self.left_corner_index, 1] = scaled_y 
        else:
            if self.right_corner_index < 4:
                self.right_corners[self.right_corner_index, 0] = scaled_x 
                self.right_corners[self.right_corner_index, 1] = scaled_y 
                self.right_corner_index += 1
            else:
                self.right_corner_index = 0
                self.right_corners[self.right_corner_index, 0] = scaled_x 
                self.right_corners[self.right_corner_index, 1] = scaled_y 
        if self.left_corner_index == 4 and self.right_corner_index == 4:
            self.H_matrix = algos.calculate_homography(self.left_corners, self.right_corners)
            points = self.H_matrix @ np.concatenate([self.right_corners, np.ones([4, 1])], axis=-1).T
            self.backprojected_corners = (points.T)[:4, :2] / (points.T)[:4, 2:]

        self.dirty_canvas = True

        self.print_current_state()

if __name__ == '__main__':
    app.use(base_app.get_backend_api())
    parser = viscomp.parse_options()
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--image-path', type=str, required=True,
                           help='Path to the image to use for the assignment')
    app_group.add_argument('--reference-solution', action='store_true',
                           help='Set this to use the precompiled binary to run the reference solution.') 
    args = parser.parse_args()
    
    import sourcedefender
    if args.reference_solution:
        import viscomp.binaries as algos
    else:
        import viscomp.algos as algos

    window = Assignment1App(args)
    app.run()
