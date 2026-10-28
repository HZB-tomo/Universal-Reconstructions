"""
imageviewer: a reusable Qt image viewer for stacks of 2D images, built from separate pieces.
It depends only on numpy, scipy, pyqtgraph and PySide6 -- nothing of the rest of this
repository -- so it can be copied into another project as a folder.

    canvas.py     ImageCanvas       one frame with pan/zoom, contrast levels, colormap
    stack.py      StackViewer       canvas + slider + z/y/x slicing + cursor readout (the usual entry point)
    histogram.py  HistogramPanel    contrast control; dock it under the image
    tools.py      ToolController    drawing tools and their toolbar (line, ...)
    profile.py    LineProfileWindow live line profile (K)
    measure.py    line length / angle / profile arithmetic, no Qt
    slicing.py    z / y / x section helpers, no Qt

    viewer = StackViewer()
    viewer.show_stack(volume)
    main_window.setCentralWidget(viewer)
    main_window.addToolBar(viewer.make_toolbar())
    dock.setWidget(viewer.make_histogram())      # a QDockWidget at the bottom
"""
from imageviewer.canvas import ImageCanvas
from imageviewer.histogram import HistogramPanel
from imageviewer.stack import StackViewer
from imageviewer.tools import LineTool, Tool, ToolController

__all__ = ["ImageCanvas", "HistogramPanel", "StackViewer", "Tool", "LineTool", "ToolController"]
