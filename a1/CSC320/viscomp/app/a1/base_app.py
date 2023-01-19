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
from abc import abstractmethod, ABC
import cv2

import viscomp.ops.image as img_ops
import viscomp.algos

if sys.platform == 'darwin':
    backend_api = "glfw"
else:
    backend_api = "glfw_imgui"

backend = f"glumpy.app.window.backends.backend_{backend_api}"
try:
    importlib.import_module(backend)
except:
    print(f"WARNING: Failed to import {backend_api}. You might be using an old version of glumpy. "
           "Falling back on glfw.")
    backend_api = "glfw"
    backend = f"glumpy.app.window.backends.backend_{backend_api}"

def get_backend_api():
    return backend_api

class BaseAssignmentApp(sys.modules[backend].Window, ABC):
    """This is the assignment app.

    This app uses Glumpy, which is a library that makes it simple to interface between OpenGL and 
    NumPy. For more information, look at: https://glumpy.github.io
    """

    def __init__(self, args):
        """Initializes the app. Most of this is boilerplate code that you shouldn't worry about.
        """
        self.args = args

        config = app.configuration.get_default()
        #config.profile = 'core'
        #config.major_version = 3
        #config.minor_version = 2
        if sys.platform == 'darwin':
            self.app_resolution = (720, 1024)
        else:
            self.app_resolution = (1080, 1920)
        self.buffer = np.zeros([*self.app_resolution, 4])
        self.init_state()
        
        super().__init__(width=self.app_resolution[1], height=self.app_resolution[0], 
                         fullscreen=False, config=app.configuration.get_default())

        screen_vertex = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        } """
        
        screen_fragment = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        }"""

        self.screen = gloo.Program(screen_vertex, screen_fragment, count=4, version='120')
        self.screen['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
        self.screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
        self.screen['scale'] = 1.0
        self.screen['tex'] = self.buffer

        raster_vertex = """
        uniform vec4   global_color;    // Global color
        attribute vec4 color;    // Vertex color
        attribute vec2 position;        // Vertex coordinates
        varying vec4   v_color;         // Interpolated fragment color (out)
        void main()
        {
            v_color = global_color * color;
            gl_Position = vec4(position, 0.0, 1.0);
            gl_PointSize = 20.0;
        }"""
        
        raster_fragment = """
        varying vec4   v_color;         // Interpolated fragment color (in)
        void main()
        {
            gl_FragColor = v_color;
        }"""

        self.raster = gloo.Program(raster_vertex, raster_fragment)

    @abstractmethod
    def init_state(self):
        pass

    def clear_raster_primitives(self):
        """Removes all instantiated raster primitives.
        """
        self.raster_line_buffers = []
        self.raster_point_buffers = []

    def add_raster_line(self, origin, destination, color, line_width=1):
        vertex_buffer = np.zeros(2, [("position", np.float32, 2),
                                     ("color", np.float32, 4)])
        vertex_buffer["position"] = np.array([origin, destination])
        vertex_buffer["color"] = np.array([color, color])
        vertex_buffer = vertex_buffer.view(gloo.VertexBuffer)

        index_buffer = np.array([0,1], dtype=np.uint32)
        index_buffer = index_buffer.view(gloo.IndexBuffer)

        self.raster_line_buffers.append([vertex_buffer, index_buffer, line_width])

    def add_raster_lines(self, origins, destinations, colors, line_width=1):
        vertex_buffer = np.zeros(2 * origins.shape[0], [("position", np.float32, 2),
                                                        ("color", np.float32, 4)])
        vertex_buffer["position"] = np.concatenate([origins, destinations], axis=-1).reshape(-1, 2)
        vertex_buffer["color"] = np.concatenate([colors, colors], axis=-1).reshape(-1, 4)
        vertex_buffer = vertex_buffer.view(gloo.VertexBuffer)

        index_buffer = np.arange(0, 2 * origins.shape[0]).astype(np.uint32)
        index_buffer = index_buffer.view(gloo.IndexBuffer)

        self.raster_line_buffers.append([vertex_buffer, index_buffer, line_width])
    
    def add_raster_point(self, origin, color, point_size=2):
        vertex_buffer = np.zeros(1, [("position", np.float32, 2),
                                     ("color", np.float32, 4)])
        vertex_buffer["position"] = np.array([origin])
        vertex_buffer["color"] = np.array([color])
        vertex_buffer = vertex_buffer.view(gloo.VertexBuffer)

        self.raster_point_buffers.append([vertex_buffer, point_size])

    def add_raster_points(self, origins, colors, point_size=2):
        vertex_buffer = np.zeros(origins.shape[0], [("position", np.float32, 2),
                                                    ("color", np.float32, 4)])
        vertex_buffer["position"] = np.array(origins)
        vertex_buffer["color"] = np.array(colors)
        vertex_buffer = vertex_buffer.view(gloo.VertexBuffer)

        self.raster_point_buffers.append([vertex_buffer, point_size])

    @abstractmethod
    def draw_imgui(self):
        pass
    
    def draw_raster(self):
        tex = self.screen['tex']
        h,w = tex.shape[:2]
        
        self.raster['global_color'] = 1, 1, 1, 1
        
        for raster_line_buffer in self.raster_line_buffers:
            vertex_buffer, index_buffer, line_width = raster_line_buffer
            gl.glLineWidth(line_width)
            self.raster.bind(vertex_buffer)
            self.raster.draw(gl.GL_LINES, index_buffer)

        for raster_point_buffer in self.raster_point_buffers:
            vertex_buffer, point_size = raster_point_buffer
            gl.glPointSize(point_size)
            self.raster.bind(vertex_buffer)
            self.raster.draw(gl.GL_POINTS)

    @abstractmethod
    def render(self):
        pass

    def on_draw(self, dt):
        self.set_title("Assignment App")
        
        self.clear_raster_primitives()
        
        if backend_api == "glfw_imgui":
            self.draw_imgui()

        self.render()
        
        # draw to screen
        self.clear()
        
        self.screen['tex'] = self.buffer
        self.screen.draw(gl.GL_TRIANGLE_STRIP)
        self.draw_raster()
