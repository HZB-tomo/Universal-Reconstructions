# The image viewer (`imageviewer/`)

A reusable Qt viewer for stacks of 2D images. It depends only on numpy, scipy, pyqtgraph and
PySide6, nothing else in this repository (a test checks that), so the folder can be copied
into another project. The application's own `gui/viewer.py` is a thin subclass that adds the
one reconstruction-specific thing, the crop rectangle.

## Pieces

| File | What it is | Edit it when you want to change... |
|---|---|---|
| `canvas.py` | `ImageCanvas`: one frame, pan/zoom, contrast levels, colormap, mouse hover | how the picture is drawn |
| `stack.py` | `StackViewer`: canvas + frame slider + z / y / x slicing + cursor readout | browsing a stack |
| `histogram.py` | `HistogramPanel`: the contrast control | the histogram and its buttons |
| `tools.py` | `ToolController`, `Tool`, `LineTool`, the toolbar | drawing tools |
| `profile.py` | `LineProfileWindow`: the live line profile | the profile window |
| `measure.py` | length, angle, profile along a line (no Qt) | measurement arithmetic |
| `slicing.py` | z / y / x section helpers (no Qt) | slicing conventions |

The pieces talk through the canvas's properties and signals (`frame`, `levels`, `levels_locked`,
`visible_frame()`, `frame_changed`, `levels_changed`, `view_changed`, `hovered`, `key_pressed`),
never through each other's internals, so a piece can be replaced or removed alone.

## Using it

```python
from imageviewer import StackViewer

viewer = StackViewer()
viewer.show_stack(volume)                      # or show_stack_lazy(getter, n, shape=...) / show_image(array)
window.setCentralWidget(viewer)
window.addToolBar(viewer.make_toolbar())       # a QToolBar: drag it to any edge, or float it
dock = QDockWidget("Histogram")
dock.setWidget(viewer.make_histogram())        # a panel; the dock can sit under the image or float away
window.addDockWidget(Qt.BottomDockWidgetArea, dock)
viewer.set_pixel_size(2.5, "um")               # optional: measurements also read in um
```

## The histogram

Whole image or only the visible area, linear or log counts, the black and white points as a draggable
region and as numbers. **Auto** takes percentiles (the clip box says how many percent at each end are
ignored; it removes hot pixels), **Min-Max** the full range, **Scale to view** fits the levels to what you are
looking at, **Lock** keeps the levels while you scroll through slices or load another image. Colormap and Invert
are there too, and the statistics of what the histogram shows are printed under it.

## The line tool

Choose **Line** in the toolbar and drag on the image. The label shows the length (in pixels, and in the physical
unit when the pixel size is known) and the angle: 0 degrees points right, positive turns counter-clockwise as you
see it on screen. Drag either end afterwards to adjust. One line exists at a time.

Keys while the cursor is over the image:

| Key | Does |
|---|---|
| **K** | opens the live line profile: the values along the line, redrawn whenever you move the line or change slice. "px wide" averages parallel lines across the width; Copy data puts the profile on the clipboard |
| Delete | removes the line |
| Esc | back to panning |

In a sinogram or other section where a pixel is not a length (an angle axis), only pixels are shown.

## Adding a tool

Subclass `Tool` in `tools.py` (a `name`, a `label`, a `drag(ev, view)` that receives the left-button drag),
add it to the dictionary in `ToolController.__init__`, and it appears in the toolbar. `LineTool` is the model.
