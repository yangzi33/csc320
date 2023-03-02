from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.uix.widget import Widget
from kivy.base import runTouchApp
from array import array
from kivy.core.window import Window


# create a 64x64 texture, defaults to rgb / ubyte
texture = Texture.create(size=(1280, 1024), colorfmt='rgb')

# create 64x64 rgb tab, and fill with values from 0 to 255
# we'll have a gradient from black to white
size = 1280 * 1024 * 3
buf = [int(x * 255 / size) for x in range(size)]

# then, convert the array to a ubyte string
arr = array('B', buf)
# buf = b''.join(map(chr, buf))

# then blit the buffer
texture.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')

# that's all ! you can use it in your graphics now :)
# if self is a widget, you can do this
root = Widget()
with root.canvas:
    Rectangle(texture=texture, pos=(0, 0), size=(1280*3, 1024*3))

runTouchApp(root)
