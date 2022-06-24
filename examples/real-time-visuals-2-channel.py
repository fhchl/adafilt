import sys
import threading
from itertools import cycle

import numpy as np
import sounddevice as sd
from adafilt import MultiChannelBlockLMS
from vispy import app, scene
from vispy.scene.visuals import Text
from vispy.scene.widgets import Grid, ViewBox

print(sd.query_devices())
input_device = 6
output_device = 6
device = sd.query_devices(input_device)
print(device)
samplerate = device["default_samplerate"]
print(samplerate)
latency = "low"
blocksize = 2048
T = 5
blocks = int((T * samplerate) // blocksize)
length = blocksize * 4
dtype = "float32"
channels = inch, outch = (1, 2)
signal = cycle(
    np.stack(
        (
            np.random.normal(size=blocks * blocksize).reshape(-1, blocksize),
            np.random.normal(size=blocks * blocksize).reshape(-1, blocksize),
        ),
        axis=1,
    )
)
buffersize = 20
filt = MultiChannelBlockLMS(
    Nin=2,
    length=length,
    blocklength=blocksize,
    leakage=0.99999999,
    stepsize=0.1,
    constrained=True,
)

i = 0



class FilterMonitor:
    def __init__(self):
        # vertex positions of data to draw
        N = length
        self.pos1 = np.zeros((N, 2))
        self.pos2 = np.zeros((N, 2))
        self.pos1[:, 0] = np.arange(N) / samplerate
        self.pos2[:, 0] = np.arange(N) / samplerate

        # color array
        self.color1 = (1, 0, 0)
        self.color2 = (0, 1, 0)

        canvas = scene.SceneCanvas(keys="interactive", show=True)
        main_grid = canvas.central_widget.add_grid()

        grid = Grid(border_color="r")
        info = scene.widgets.ViewBox(border_color="b")
        info.camera = "panzoom"
        info.camera.rect = -1, -1, 2, 2
        info.stretch = (1, 0.1)

        main_grid.add_widget(grid, row=0, col=0)
        main_grid.add_widget(info, row=1, col=0)

        # add some axes
        x_axis = scene.AxisWidget(orientation="bottom")
        x_axis.stretch = (1, 0.1)

        y_axis = scene.AxisWidget(orientation="left")
        y_axis.stretch = (0.1, 1)

        grid.add_widget(x_axis, row=1, col=1)
        grid.add_widget(y_axis, row=0, col=0)

        viewbox = grid.add_view(row=0, col=1, camera="panzoom")
        x_axis.link_view(viewbox)
        y_axis.link_view(viewbox)

        # add cpu information
        text = Text(text="TEXT", color=(1, 1, 1, 1), parent=info.scene)
        text.font_size = 18
        self.text = text
        # add a line plot inside the viewbox
        self.line1 = scene.Line(self.pos1, self.color1, parent=viewbox.scene)
        self.line2 = scene.Line(self.pos2, self.color2, parent=viewbox.scene)

        # auto-scale to see the whole line.
        viewbox.camera.set_range()

    def update(self, w, load):
        self.pos1[:, 1] = w[:, 0, 0]
        self.pos2[:, 1] = w[:, 0, 1]
        self.line1.set_data(pos=self.pos1, color=self.color1)
        self.line2.set_data(pos=self.pos2, color=self.color2)
        self.text.text = f"CPU load: {load * 100:.1f}%"


filter_monitor = FilterMonitor()

e = np.zeros((blocksize, 1))
y = np.zeros((blocksize, 1))


def callback(indata, outdata, frames, time, status):
    global e, y
    if status:
        print("Callback status:", status)

    try:
        u = next(signal).T  # signal
        # play signal
        outdata[:] = u

        # get system output
        d = indata[:, 0:1]

        # filter prediction
        y[:] = filt.filt(u)

        # error signal
        e[:] = d - y

        # weight adaptation
        filt.adapt(u[:, None, None], e)

        filter_monitor.update(filt.w, stream.cpu_load)

    except StopIteration:
        raise sd.CallbackAbort
    except Exception as e:
        print(type(e).__name__ + ": " + str(e))
        raise sd.CallbackAbort

callback_finished_event = threading.Event()

stream = sd.Stream(
    device=(input_device, output_device),
    samplerate=samplerate,
    blocksize=blocksize,
    dtype=dtype,
    latency=latency,
    channels=channels,
    callback=callback,
    finished_callback=callback_finished_event.set,
)

try:
    with stream:
        app.run()
        # callback_finished_event.wait()
except KeyboardInterrupt:
    sys.exit(0)

# plt.plot(filt.w)
# plt.show()
