from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QPen

# class ClickableLabel(QLabel):
#     pressed = pyqtSignal(int, int)  # Custom signal emitting x, y coordinates for press
#     released = pyqtSignal(int, int)  # Custom signal emitting x, y coordinates for release

#     def __init__(self, parent=None):
#         super(ClickableLabel, self).__init__(parent)
#         self.setAlignment(Qt.AlignCenter)
#         self.setMouseTracking(True)

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.pressed.emit(event.pos().x(), event.pos().y())

#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.released.emit(event.pos().x(), event.pos().y())
            
class ClickableLabel(QLabel):
    rectangleCompleted = pyqtSignal(QRect)  # Signal to emit the rectangle coordinates

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.original_pixmap = None
        self.press_position = None
        self.current_position = None
        self.on_place = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.press_position = event.pos()
            self.current_position = event.pos()
            self.original_pixmap = self.pixmap().copy()  # Store the current pixmap safely

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.current_position = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.current_position = event.pos()
            if self.press_position == self.current_position:
                self.on_place = True
            # self.drawing = False
            self.update()
            self.emitRectangle()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.on_place:
            self.emitRectangle()
            self.on_place = False
            
        elif self.press_position and self.current_position:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.green, 3, Qt.SolidLine))
            rect = QRect(self.press_position, self.current_position)
            painter.drawRect(rect.normalized())  # Ensure the rectangle is correctly oriented

    def emitRectangle(self):
        if self.press_position and self.current_position:
            rect = QRect(self.press_position, self.current_position)
            self.rectangleCompleted.emit(rect.normalized())  # Emit the normalized rectangle

    def clearDrawing(self):
        self.press_position = None
        self.current_position = None
        self.setPixmap(self.original_pixmap)  # Reset to original pixmap if needed
