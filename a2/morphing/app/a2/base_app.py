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
from abc import abstractmethod, ABC
import cv2

import viscomp.ops.image as img_ops
import viscomp.algos
import math

from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Point, Line, InstructionGroup, Color
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.dropdown import DropDown
from kivy.base import runTouchApp
from array import array
from kivy.core.window import Window
from kivy.clock import Clock

class BaseAssignmentRunner(App):
    def __init__(self, window):
        super().__init__()
        self.window = window
    
    def build(self):
        Clock.schedule_interval(self.window.on_draw, 1.0 / 30.0)
        return self.window

class BaseAssignmentApp(Widget):
    """This is the assignment app.
    """

    def __init__(self, args):
        """Initializes the app. Most of this is boilerplate code that you shouldn't worry about.
        """
        super().__init__()
        
        self.set_app_resolution()
        self.args = args
        
        h, w = self.app_resolution
        self.buffer = np.zeros([h, w, 4])
        
        if sys.platform == 'darwin':
            Window.size = (w//2 + 200, h//2)
            offset = 400
        else:
            Window.size = (w + 200, h)
            offset = 200
        Window.bind(mouse_pos=self.on_mouse_pos, on_key_down=self.on_key_down)

        self.texture = Texture.create(size=(w, h), colorfmt='rgb')
        
        self.buffer[:, w//2 - 2 : w//2 + 2, :] = 1.0
        self.texture.blit_buffer((self.buffer[..., :3]*255).astype(np.uint8).tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.init_state()
        self.objects = []

        with self.canvas:
            Rectangle(texture=self.texture, pos=(0, 0), size=(self.app_resolution[1], self.app_resolution[0]))
            Color(0.9, 0.9, 0.9)
            Rectangle(pos=(w, 0), size=(offset, h))

        ### Create menu
        menu = BoxLayout(pos=(w, 0), size=(offset, h), orientation='vertical')
        #widget = StackLayout(orientation='horizontal')
        
        widget = Button(text="Run Warp (r)", size_hint=(1.0, 0.5))
        def callback(instance):
            self.dirty_canvas = True
        widget.bind(on_press=callback)
        menu.add_widget(widget)
        
        widget = Button(text="Clear Arrows (c)", size_hint=(1.0, 0.5))
        def callback(instance):
            self.clear_arrows()
        widget.bind(on_press=callback)
        menu.add_widget(widget)
        
        widget = Button(text="Export Arrows and Image (e)", size_hint=(1.0, 0.5))
        def callback(instance):
            self.export()
        widget.bind(on_press=callback)
        menu.add_widget(widget)
        
        widget = Button(text="Export Frames", size_hint=(1.0, 0.5))
        def callback(instance):
            self.export_video()
        widget.bind(on_press=callback)
        menu.add_widget(widget)
        
        widget = StackLayout()
        slider = Slider(min=1, max=15, value=self.num_frames//10, size_hint=(0.6, 0.45),cursor_size=(20, 20),
                value_track=True, value_track_color=[1, 0, 0, 1])
        text_frame = Label(text=f"{self.num_frames}", color=(0,0,0), size_hint=(0.3, 0.25))
        def callback_frame(instance, value):
            self.num_frames = int(round(value) * 10)
            text_frame.text = f"{self.num_frames}"
        slider.bind(value=callback_frame)
        widget.add_widget(slider)
        widget.add_widget(Label(text="num_frames", color=(0,0,0), size_hint=(0.3, 0.45)))
        widget.add_widget(text_frame)
        menu.add_widget(widget)
        
        widget = BoxLayout(orientation="vertical")
        widget.add_widget(Label(text="Left Image Buffer", color=(0,0,0), size_hint=(0.7, 0.7)))
        row = BoxLayout(orientation="horizontal")
        btn1 = ToggleButton(text='S', group='left', state='down')
        btn2 = ToggleButton(text='WD', group='left')
        btn3 = ToggleButton(text='M', group='left')
        def callback(instance):
            self.left_buffer = instance.text
            self.update_buffer()
        btn1.bind(on_press=callback)
        btn2.bind(on_press=callback)
        btn3.bind(on_press=callback)
        row.add_widget(btn1)
        row.add_widget(btn2)
        row.add_widget(btn3)
        widget.add_widget(row)
    
        menu.add_widget(widget)
        
        widget = BoxLayout(orientation="vertical")
        widget.add_widget(Label(text="Right Image Buffer", color=(0,0,0), size_hint=(0.7, 0.7)))
        row = BoxLayout(orientation="horizontal")
        btn1 = ToggleButton(text='D', group='right')
        btn2 = ToggleButton(text='WS', group='right')
        btn3 = ToggleButton(text='M', group='right')
        if self.args.destination_image_path:
            btn1.state = 'down'
        else:
            btn2.state = 'down'
        def callback(instance):
            self.right_buffer = instance.text
            self.update_buffer()
        btn1.bind(on_press=callback)
        btn2.bind(on_press=callback)
        btn3.bind(on_press=callback)
        row.add_widget(btn1)
        row.add_widget(btn2)
        row.add_widget(btn3)
        widget.add_widget(row)
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.edit_arrows, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.edit_arrows = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Edit Arrows", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.show_warped_cursor, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.show_warped_cursor = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Show Warped Cursor", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.show_source_morph_lines, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.show_source_morph_lines = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Show Source Lines", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.show_destination_morph_lines, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.show_destination_morph_lines = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Show Dest. Lines", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.run_realtime, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.run_realtime = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Run Realtime", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.verbose, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.verbose = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Verbose Mode", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.bilinear, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.bilinear = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Bilinear", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.vectorize, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.vectorize = value
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Vectorize", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)
        
        widget = StackLayout()
        checkbox = CheckBox(active=self.debug_grid, size_hint=(0.15, 0.45), color=(0,0,0))
        def callback(instance, value):
            self.debug_grid = value
            self.draw_image()
        checkbox.bind(active=callback)
        widget.add_widget(checkbox)
        widget.add_widget(Label(text="Show Grid", color=(0,0,0), size_hint=(0.7, 0.45)))
        menu.add_widget(widget)

        widget = StackLayout()
        slider = Slider(min=1, max=7, value=self.supersampling, size_hint=(0.6, 0.25),cursor_size=(20, 20),
                value_track=True, value_track_color=[1, 0, 0, 1])
        text_patch = Label(text=f"{self.supersampling}", color=(0,0,0), size_hint=(0.3, 0.25))
        def callback_patch(instance, value):
            self.supersampling = int(round(value))
            text_patch.text = f"{self.supersampling}"
        slider.bind(value=callback_patch)
        widget.add_widget(slider)
        widget.add_widget(Label(text="supersampling", color=(0,0,0), size_hint=(0.3, 0.25)))
        widget.add_widget(text_patch)
        menu.add_widget(widget)
        
        widget = StackLayout()
        slider = Slider(min=1e-4, max=2.0, value=self.param_a, size_hint=(0.6, 0.25),cursor_size=(20, 20),
                value_track=True, value_track_color=[1, 0, 0, 1])
        text_a = Label(text=f"{self.param_a:.4f}", color=(0,0,0), size_hint=(0.3, 0.25))
        def callback_a(instance, value):
            text_a.text = f"{value:.4f}"
            self.param_a = value
        slider.bind(value=callback_a)
        widget.add_widget(slider)
        widget.add_widget(Label(text="param a", color=(0,0,0), size_hint=(0.3, 0.25)))
        widget.add_widget(text_a)
        menu.add_widget(widget)
        
        widget = StackLayout()
        slider = Slider(min=0.5, max=2.0, value=self.param_b, size_hint=(0.6, 0.25), cursor_size=(20, 20),
                value_track=True, value_track_color=[1, 0, 0, 1])
        text_b = Label(text=f"{self.param_b:.4f}", color=(0,0,0), size_hint=(0.3, 0.25))
        def callback_b(instance, value):
            text_b.text = f"{value:.4f}"
            self.param_b = value
        slider.bind(value=callback_b)
        widget.add_widget(slider)
        widget.add_widget(Label(text="param b", color=(0,0,0), size_hint=(0.3, 0.25)))
        widget.add_widget(text_b)
        menu.add_widget(widget)
        
        widget = StackLayout()
        slider = Slider(min=0.0, max=1.0, value=self.param_p, size_hint=(0.6, 0.25),cursor_size=(20, 20),
                value_track=True, value_track_color=[1, 0, 0, 1])
        text_p = Label(text=f"{self.param_p:.4f}", color=(0,0,0), size_hint=(0.3, 0.25))
        def callback_p(instance, value):
            text_p.text = f"{value:.4f}"
            self.param_p = value
        slider.bind(value=callback_p)
        widget.add_widget(slider)
        widget.add_widget(Label(text="param p", color=(0,0,0), size_hint=(0.3, 0.25)))
        widget.add_widget(text_p)
        menu.add_widget(widget)

        widget = StackLayout()
        slider = Slider(min=0.0, max=1.0, value=self.param_t, size_hint=(0.6, 0.25),cursor_size=(20, 20),
                value_track=True, value_track_color=[1, 0, 0, 1])
        text_t = Label(text=f"{self.param_t:.4f}", color=(0,0,0), size_hint=(0.3, 0.25))
        def callback(instance, value):
            text_t.text = f"{value:.4f}"
            self.param_t = value
        slider.bind(value=callback)
        widget.add_widget(slider)
        widget.add_widget(Label(text="param t", color=(0,0,0), size_hint=(0.3, 0.25)))
        widget.add_widget(text_t)
        menu.add_widget(widget)
        
        self.add_widget(menu)
        
    def on_touch_down(self, touch):
        super().on_touch_down(touch)
        h, w = self.app_resolution
        self.on_mouse_press(touch.pos[0], h-touch.pos[1], None)
    
    def on_touch_move(self, touch):
        super().on_touch_move(touch)
        h, w = self.app_resolution
        self.on_mouse_drag(touch.pos[0], h-touch.pos[1], None, None, None)
    
    def on_touch_up(self, touch):
        super().on_touch_up(touch)
        h, w = self.app_resolution
        self.on_mouse_release(touch.pos[0], h-touch.pos[1], None)
    
    def on_mouse_pos(self, window, pos):
        h, w = self.app_resolution
        self.on_mouse_motion(pos[0], h-pos[1], None, None)

    def on_key_down(self, keyboard, keycode, text, modifiers, _):
        self.on_key_press(int(keycode)-32, None)

    def set_app_resolution(self):
        """Sets the app resolution.
        """
        if sys.platform == 'darwin':
            #self.app_resolution = (720*2, 1024*2)
            self.app_resolution = (720, 1024)
        else:
            self.app_resolution = (720, 1024)
    
    def init_state(self):
        pass

    def clear_raster_primitives(self):
        """Removes all instantiated raster primitives.
        """
        self.raster_line_buffers = []
        self.raster_point_buffers = []
        for obj in self.objects:
            self.canvas.remove(obj)
        self.objects = []

    def add_raster_line(self, origin, destination, color, line_width=1):
        vertex_buffer = np.zeros(2, [("position", np.float32, 2),
                                     ("color", np.float32, 4)])
        vertex_buffer["position"] = np.array([origin, destination])
        vertex_buffer["color"] = np.array([color, color])
        
        index_buffer = np.array([0,1], dtype=np.uint32)
        self.raster_line_buffers.append([vertex_buffer, index_buffer, line_width])

    def add_raster_lines(self, origins, destinations, colors, line_width=1):
        vertex_buffer = np.zeros(2 * origins.shape[0], [("position", np.float32, 2),
                                                        ("color", np.float32, 4)])
        vertex_buffer["position"] = np.concatenate([origins, destinations], axis=-1).reshape(-1, 2)
        vertex_buffer["color"] = np.concatenate([colors, colors], axis=-1).reshape(-1, 4)
        index_buffer = np.arange(0, 2 * origins.shape[0]).astype(np.uint32)
        self.raster_line_buffers.append([vertex_buffer, index_buffer, line_width])
    
    def add_raster_point(self, origin, color, point_size=2):
        vertex_buffer = np.zeros(1, [("position", np.float32, 2),
                                     ("color", np.float32, 4)])
        vertex_buffer["position"] = np.array([origin])
        vertex_buffer["color"] = np.array([color])
        self.raster_point_buffers.append([vertex_buffer, point_size])

    def add_raster_points(self, origins, colors, point_size=2):
        vertex_buffer = np.zeros(origins.shape[0], [("position", np.float32, 2),
                                                    ("color", np.float32, 4)])
        vertex_buffer["position"] = np.array(origins)
        vertex_buffer["color"] = np.array(colors)
        self.raster_point_buffers.append([vertex_buffer, point_size])

    def draw_raster(self):
        for raster_line_buffer in self.raster_line_buffers:
            vertex_buffer, index_buffer, line_width = raster_line_buffer
            pixel_positions = img_ops.unnormalize_coordinates(vertex_buffer["position"], *self.app_resolution)
            obj = InstructionGroup()
            color = vertex_buffer["color"][0,:3]
            obj.add(Color(*list(color)))
            obj.add(Line(points=list(pixel_positions.reshape(-1)), width=line_width/2))
            self.objects.append(obj)
            self.canvas.add(obj)

        for raster_point_buffer in self.raster_point_buffers:
            vertex_buffer, point_size = raster_point_buffer
            pixel_positions = img_ops.unnormalize_coordinates(vertex_buffer["position"], *self.app_resolution)
            obj = InstructionGroup()
            color = vertex_buffer["color"][0,:3]
            obj.add(Color(*list(color)))
            obj.add(Point(points=list(pixel_positions.reshape(-1)), pointsize=point_size*2))
            self.objects.append(obj)
            self.canvas.add(obj)

    def render(self):
        pass

    def on_draw(self, dt):
        # draw to screen
        self.clear_raster_primitives()

        self.render()
        self.texture.blit_buffer((self.buffer[..., :3]*255).astype(np.uint8).tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.draw_raster()
