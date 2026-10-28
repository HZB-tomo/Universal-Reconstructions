"""
Tools you draw with on the image, and the toolbar that chooses between them.

    controller = ToolController(canvas)
    window.addToolBar(controller.make_toolbar())      # a QToolBar: dockable, floatable

A Tool receives the left-button drags on the canvas while it is active (Tool.drag).
LineTool draws a measurement line with its length and angle; its ends can be dragged
afterwards. Keys while the cursor is over the image: K opens the live line profile,
Delete removes the line, Escape goes back to panning.

To add a tool: subclass Tool, implement drag(), register it in ToolController.__init__.
"""
from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QToolBar

from imageviewer.canvas import ImageCanvas
from imageviewer.measure import describe_line, line_length

MIN_DRAG_PIXELS = 2.0        # a click or a tiny drag does not make a line


class LineMeasurement(QObject):
    """One measurement line on the canvas, with its length/angle label."""

    changed = Signal()

    def __init__(self, canvas: ImageCanvas, p0, p1):
        super().__init__(canvas)
        self.canvas = canvas
        pen = pg.mkPen(0, 220, 255, width=2)
        self.roi = pg.LineSegmentROI([tuple(p0), tuple(p1)], pen=pen, hoverPen=pg.mkPen(255, 220, 0, width=2))
        self.label = pg.TextItem(anchor=(0, 1), color=(255, 255, 255), fill=(0, 0, 0, 150))
        self.label.setZValue(20)
        canvas.view.addItem(self.roi)
        canvas.view.addItem(self.label)
        self.roi.sigRegionChanged.connect(self._on_moved)
        self.refresh_label()

    @property
    def points(self) -> tuple:
        """((x0, y0), (x1, y1)) in view coordinates."""
        a, b = (self.roi.mapToParent(h.pos()) for h in self.roi.endpoints)
        return (a.x(), a.y()), (b.x(), b.y())

    def set_end(self, p) -> None:
        self.roi.movePoint(self.roi.endpoints[1], pg.Point(p[0], p[1]), finish=False)

    @property
    def length(self) -> float:
        return line_length(*self.points)

    def refresh_label(self) -> None:
        p0, p1 = self.points
        self.label.setText(describe_line(p0, p1, self.canvas.physical_pixel()))
        self.label.setPos(*p1)

    def _on_moved(self) -> None:
        self.refresh_label()
        self.changed.emit()

    def remove(self) -> None:
        self.canvas.view.removeItem(self.roi)
        self.canvas.view.removeItem(self.label)
        self.roi.sigRegionChanged.disconnect(self._on_moved)


class Tool:
    name = ""
    label = ""
    tooltip = ""

    def __init__(self, controller: "ToolController"):
        self.controller = controller

    def activated(self) -> None:
        pass

    def deactivated(self) -> None:
        pass

    def drag(self, ev, view) -> None:
        """A left-button drag on the canvas: pyqtgraph's mouse drag event, and the ViewBox it happened in."""


class LineTool(Tool):
    name = "line"
    label = "Line"
    tooltip = "Draw a line: drag on the image. Shows its length and angle; drag its ends to adjust. Press K for the profile."

    def __init__(self, controller):
        super().__init__(controller)
        self._drawing: Optional[LineMeasurement] = None

    def drag(self, ev, view) -> None:
        if ev.isStart():
            start = view.mapToView(ev.buttonDownPos())
            self.begin((start.x(), start.y()))
        point = view.mapToView(ev.pos())
        self.move_to((point.x(), point.y()))
        if ev.isFinish():
            self.finish()

    # The three steps of a drag, public so that they can be driven without a mouse.

    def begin(self, p) -> None:
        self._drawing = self.controller.new_measurement(p, p)

    def move_to(self, p) -> None:
        if self._drawing is not None:
            self._drawing.set_end(p)

    def finish(self) -> None:
        line, self._drawing = self._drawing, None
        if line is not None and line.length < MIN_DRAG_PIXELS:
            self.controller.discard(line)


class ToolController(QObject):
    active_tool_changed = Signal(object)     # the tool's name, or None for plain panning
    measurement_changed = Signal()           # the current line moved, appeared or went away

    def __init__(self, canvas: ImageCanvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.tools = {tool.name: tool for tool in (LineTool(self),)}
        self.active: Optional[Tool] = None
        self.current: Optional[LineMeasurement] = None
        self._profile = None
        canvas.key_pressed.connect(self._on_key)
        canvas.frame_changed.connect(self._refresh_label)

    # -- choosing a tool -----------------------------------------------------

    def set_active(self, name: Optional[str]) -> None:
        tool = self.tools.get(name) if name else None
        if tool is self.active:
            return
        if self.active is not None:
            self.active.deactivated()
        self.active = tool
        self.canvas.view.active_tool = tool
        if tool is not None:
            tool.activated()
        self.active_tool_changed.emit(tool.name if tool else None)

    # -- measurements ------------------------------------------------------------

    def new_measurement(self, p0, p1) -> LineMeasurement:
        """Start a line, replacing the current one (there is one measurement line at a time)."""
        self.clear()
        line = LineMeasurement(self.canvas, p0, p1)
        line.changed.connect(self.measurement_changed)
        self.current = line
        self.measurement_changed.emit()
        return line

    def discard(self, line: LineMeasurement) -> None:
        if line is self.current:
            self.clear()

    def clear(self) -> None:
        if self.current is not None:
            self.current.remove()
            self.current.deleteLater()
            self.current = None
            self.measurement_changed.emit()

    def _refresh_label(self) -> None:
        if self.current is not None:
            self.current.refresh_label()           # the pixel size or the section may have changed

    # -- the profile window -----------------------------------------------------------

    def open_profile(self):
        from imageviewer.profile import LineProfileWindow

        if self._profile is None:
            self._profile = LineProfileWindow(self)
        self._profile.update_profile()
        self._profile.show()
        self._profile.raise_()
        self._profile.activateWindow()
        return self._profile

    # -- keys ----------------------------------------------------------------------------

    def _on_key(self, key: int) -> None:
        if key == Qt.Key_K:
            self.open_profile()
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.clear()
        elif key == Qt.Key_Escape:
            self.set_active(None)

    # -- the toolbar -------------------------------------------------------------------------

    def make_toolbar(self, title: str = "Image tools", parent=None) -> QToolBar:
        """A toolbar for the tools. Add it to a QMainWindow to make it dockable and floatable."""
        bar = QToolBar(title, parent)
        bar.setObjectName("imageviewer_tools")
        group = QActionGroup(bar)
        group.setExclusive(True)
        actions = {}

        pan = bar.addAction("Pan / zoom")
        pan.setCheckable(True)
        pan.setChecked(True)
        pan.setToolTip("Drag to pan, wheel to zoom (Esc)")
        pan.triggered.connect(lambda: self.set_active(None))
        group.addAction(pan)
        actions[None] = pan
        for tool in self.tools.values():
            action = bar.addAction(tool.label)
            action.setCheckable(True)
            action.setToolTip(tool.tooltip)
            action.triggered.connect(lambda _checked=False, name=tool.name: self.set_active(name))
            group.addAction(action)
            actions[tool.name] = action
        bar.addSeparator()
        profile = bar.addAction("Profile (K)")
        profile.setToolTip("Open the live line profile of the line")
        profile.triggered.connect(self.open_profile)
        clear = bar.addAction("Clear")
        clear.setToolTip("Remove the line (Delete)")
        clear.triggered.connect(self.clear)

        self.active_tool_changed.connect(lambda name: actions.get(name, pan).setChecked(True))
        bar.actions_by_tool = actions
        return bar
