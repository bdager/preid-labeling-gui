import os
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QSlider,
    QFileDialog,
    QMessageBox,
    QShortcut,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtCore
from PyQt5.QtGui import QKeySequence
import cv2
import json
from copy import deepcopy

from utils.savings import *
from utils.refine_dets import *

# import ui files
from ui_sources.dataset_gui import *

execution_path = os.getcwd()
video_path_dir = os.path.sep.join([execution_path, "..", "seq"])
json_path_dir = os.path.sep.join([execution_path, "..", "seq_etiquetadas"])
save_path_dir = os.path.sep.join([execution_path, "..", "seq_etiquetadas"])

# class MainWindow(QMainWindow):
# def __init__(self):
#         super().__init__()
# uic.loadUi('ui_sources/dataset_gui.ui', self)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main class containing the main window of the graphical interface."""

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # method for generating the interface
        self.initUI()
        self.cap = None
        self.frames = []
        self.frame_pos = 0
        self.fps = 0
        self.video_length = 0
        self.json_data = {}
        self.json_data_previous = []
        self.video_path = ""
        self.json_path = ""
        self.save_directory = ""
        self.pixmap = None
        self.bbox = []
        self.img_width = None
        self.img_height = None
        self.clicked_det = []
        self.save_status = True
        self.open_json_status = True
        self.colors = (np.random.rand(32, 3) * 255).astype(int)

        print("VideoPlayer Initialized")

    def initUI(self):
        # Connect the clicked signal to a custom slot
        self.label_VideoFrame.rectangleCompleted.connect(self.label_clicked)
        # self.label_VideoFrame.released.connect(self.label_clicked)

        # Buttons
        self.PB_open_video.clicked.connect(self.openVideo)
        self.PB_open_json.clicked.connect(self.openJSON)
        self.PB_play.clicked.connect(self.toggleVideoPlayback)
        self.PB_delete.clicked.connect(self.deleteId)
        self.PB_save.clicked.connect(self.saveResults)
        self.PB_replace.clicked.connect(self.changeId)
        self.PB_add.clicked.connect(self.addNewDet)
        self.PB_flickering.clicked.connect(self.correctFlickering)

        # Slider
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.changeFrame)

        # Timer
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrameSlot)

        # Set up the Ctrl+Z shortcut for undo
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.removeAnnotation)

    def openVideo(self):
        # Open video file
        self.video_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", video_path_dir, "Video Files (*.mp4 *.avi *.mov *.webm)"
        )
        if self.video_path != "":
            self.label_flickerin_status.setText("")
            self.loadVideo()

    def openJSON(self):
        # Check if the results have been saved and the json file has been opened for the first time
        if not self.save_status and not self.open_json_status:
            # Ask if the user wants to save the changes
            reply = self.show_question_message(
                "Save", "Do you want to save the changes first?"
            )
            if reply == QMessageBox.Yes:
                self.saveResults()
        # Open JSON file
        self.json_path, _ = QFileDialog.getOpenFileName(
            self, "Open JSON File", json_path_dir, "JSON Files (*.json)"
        )
        if self.json_path != "":
            self.json_data = read_json_file(self.json_path)
            self.json_data_previous.append(deepcopy(self.json_data))
            print("JSON data loaded")
            self.updateImage()
            self.open_json_status = False

    def loadVideo(self):
        self.cap = cv2.VideoCapture(self.video_path)
        # Get the total numer of frames in the video
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Frames loaded: ", self.video_length)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.slider.setMaximum(self.video_length - 1)
        self.slider.setValue(0)
        self.PB_play.setText("Play")
        self.updateImage()
        # self.cap.release()

    def nextFrameSlot(self):
        # Advance one frame
        if self.frame_pos < self.video_length - 1:
            self.frame_pos += 1
            self.updateImage()
            self.slider.setValue(self.frame_pos)
        else:
            self.timer.stop()
            self.PB_play.setText("Play")
            self.is_playing = False

    def toggleVideoPlayback(self):
        if self.is_playing:
            self.timer.stop()
            self.PB_play.setText("Play")
        else:
            if self.video_length > 0:
                self.timer.start(
                    int(1000 / self.fps)
                )  # Start the timer at the video frame rate
                self.PB_play.setText("Stop")
        self.is_playing = not self.is_playing

    def addDet(self, frame_out):
        if self.json_data[str(self.frame_pos + 1)] is not []:
            for det in self.json_data[str(self.frame_pos + 1)]:
                id_label = det["id"]
                bboxP = det["BboxP"]
                # Check if the detection has bboxF key
                if "BboxF" in det:
                    bboxF = det["BboxF"]
                    self.bboxF_status = True
                else:
                    bboxF = None
                    self.bboxF_status = False
                draw_det(bboxP, id_label, frame_out, self.colors)
                if bboxF is not None:
                    if bboxF == [-1]:
                        bboxF = []
                    if bboxF != []:
                        # print("bboxF: ", bboxF)
                        draw_det(bboxF, id_label, frame_out, self.colors)

    def readFrame(self):
        # Read an specific frame from the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
        ret, frame = self.cap.read()

        frame_out = frame.copy()

        if self.json_data != {}:
            self.addDet(frame_out)

        if ret:
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            height, width = frame_out.shape[:2]
            qImg = QImage(frame_out, width, height, QImage.Format_RGB888)
            return qImg
        else:
            return None

    def readIds(self):
        if self.json_data == {}:
            return []
        ids = []
        # Read the ids from the json file
        for det in self.json_data[str(self.frame_pos + 1)]:
            ids.append(det["id"])
            print(det)
        return ids

    def updateImage(self):
        # Update the QLself.label_VideoFrame. with the current frame
        self.pixmap = QPixmap.fromImage(self.readFrame())
        self.label_VideoFrame.setPixmap(
            self.pixmap.scaled(
                self.label_VideoFrame.width(),
                self.label_VideoFrame.height(),
                Qt.KeepAspectRatio,
            )
        )
        # self.label_VideoFrame.setPixmap(pixmap)
        self.label_frameId.setText(str(self.frame_pos + 1))
        self.label_ids.setText(str(self.readIds()))
        self.label_save.setText("")
        self.save_status = False
        self.PB_save.setEnabled(True)

    def changeFrame(self, value):
        # Change the frame on the slider move
        self.frame_pos = value
        self.label_flickerin_status.setText("")
        self.updateImage()

    def getValues(self):
        ids = []
        to_frame = None
        from_frame = None

        # Get the new id
        new_id = self.text_newId.text()
        # Check new id is an integer
        if new_id != "":
            new_id = self.convert_text_to_int(new_id)

        # Get ids to change/add/delete
        if self.text_actId.text() != "":
            ids = self.text_actId.text().split(",")
            ids = list(map(self.convert_text_to_int, ids))
            print(ids)
        else:
            # If is empty get the ids from the json file
            ids = self.readIds()
            print(ids)

        # Get the initial frame to change/delete the id in that range of frames
        if self.text_fromFrame.text() != "":
            from_frame = self.convert_text_to_int(self.text_fromFrame.text())
        else:
            from_frame = self.frame_pos + 1

        # Get the final frame to change/delete the id in that range of frames
        if self.text_toFrame.text() != "":
            to_frame = self.convert_text_to_int(self.text_toFrame.text())
        else:
            to_frame = self.frame_pos + 1

        return ids, new_id, from_frame, to_frame

    def getFlickeringValues(self):
        max_age = 5
        to_frame = None
        from_frame = None

        # Get the max_age
        if self.lineEdit_maxAge.text() != "":
            max_age = self.convert_text_to_int(self.lineEdit_maxAge.text())

        # Get the initial frame to change/delete the id in that range of frames
        if self.text_fromFrameFlick.text() != "":
            from_frame = self.convert_text_to_int(self.text_fromFrameFlick.text())
        else:
            from_frame = self.frame_pos + 1

        # Get the final frame to change/delete the id in that range of frames
        if self.text_toFrameFlick.text() != "":
            to_frame = self.convert_text_to_int(self.text_toFrameFlick.text())
        else:
            to_frame = self.frame_pos + 1

        return max_age, from_frame, to_frame

    def changeId(self):
        """Change the id of the person in the json file"""
        if self.json_data == {}:
            return

        # Do a copy of the json_data to use in Ctrl+z command
        self.save_lasts_annotations()

        ids, new_id, from_frame, to_frame = self.getValues()

        # Check if the new id is empty
        if new_id == "":
            self.show_error_message("The new id is empty")
            return

        # Change the ids in the json file
        for frame in range(int(from_frame), int(to_frame) + 1):
            # Check if a detection has been selected by clicking on it
            # and modify it
            if self.clicked_det != []:
                # Check if the detection is in the json file
                for det in self.json_data[str(frame)]:
                    if det == self.clicked_det:
                        det["id"] = new_id
                        print("Id changed")
                        break
                # If the detection is not in the json file, get the detection with the best iou
                # with a high confidence if it exists
                best_iou, best_det = calc_iou(
                    self.clicked_det, self.json_data[str(frame)], confidence=0.7
                )
                if best_iou != 0.0:
                    best_det["id"] = new_id
                    self.clicked_det = best_det
                    print("Id changed")

            # Otherwise, change ids detections in the json file based only on
            # the ids enter in the text_edit field (act_id)
            # Comment/uncomment break line to change all/one ids in the analyzed frame
            else:
                # data_copy = deepcopy(self.json_data[str(frame)])
                for det in self.json_data[str(frame)]:
                    if det["id"] in ids:
                        det["id"] = new_id
                        print("Id changed")
                        break
        # Reset the clicked_det variable
        self.clicked_det = []
        self.updateImage()

    def deleteId(self):
        """Delete the id from the json file.
        If ids are not entered in the text_edit field, the ids are taken from the json file,
        so, all the ids in the define range of frames are deleted."""
        if self.json_data == {}:
            return

        # Do a copy of the json_data to use in Ctrl+z command
        self.save_lasts_annotations()

        ids, _, from_frame, to_frame = self.getValues()

        # Delete the ids from the json file
        for frame in range(int(from_frame), int(to_frame) + 1):
            # Check if a detection has been selected by clicking on it
            # and delete it
            if self.clicked_det != []:
                # Check if the detection is in the json file
                if self.clicked_det in self.json_data[str(frame)]:
                    self.json_data[str(frame)].remove(self.clicked_det)
                    print("Id removed")
                    continue

                # If the detection is not in the json file, get the detection with the best iou
                # with a high confidence if it exists
                best_iou, best_det = calc_iou(
                    self.clicked_det, self.json_data[str(frame)], confidence=0.7
                )
                if best_iou != 0.0:
                    self.json_data[str(frame)].remove(best_det)
                    self.clicked_det = best_det
                    print("Id removed")

            # Otherwise, delete detections in the json file based only on
            # the ids enter in the text_edit field (act_id)
            else:
                data_copy = deepcopy(self.json_data[str(frame)])
                for det in data_copy:
                    if det["id"] in ids:
                        self.json_data[str(frame)].remove(det)
                        print("Id removed")

        # Reset the clicked_det variable
        self.clicked_det = []
        self.updateImage()

    def saveResults(self):
        # Select a directory to save files
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", save_path_dir
        )

        # Get the base name of the video file without the extension
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]

        json_file = base_name + ".json"
        video_file = base_name + ".avi"

        json_output_path = os.path.join(dir_path, json_file)
        video_output_path = os.path.join(dir_path, video_file)

        # Save the corrected data in a json file
        print("Saving json file...")
        saveJson(self.json_data, json_output_path)

        # Save the corrected data in a video file
        # print("Saving video file...")
        # saveVideo(self.video_path, video_output_path, self.json_data)

        self.label_save.setText("Done")
        self.PB_save.setEnabled(False)
        self.save_status = True

    def addNewDet(self):
        # Add a new detection to the json file
        if self.json_data == {}:
            return

        # Do a copy of the json_data to use in Ctrl+z command
        self.save_lasts_annotations()

        _, new_id, from_frame, to_frame = self.getValues()

        # Check if the new id is empty
        if new_id == "":
            self.show_error_message("The new id is empty")
            return

        for frame in range(int(from_frame), int(to_frame) + 1):
            if self.bboxF_status:
                new_det = {"id": int(new_id), "BboxP": self.bbox, "BboxF": []}
            else:
                new_det = {"id": int(new_id), "BboxP": self.bbox}
            self.json_data[str(frame)].append(new_det)
            print("New detection added")

        self.bbox = []
        self.label_VideoFrame.clearDrawing()
        self.updateImage()

    def correctFlickering(self):
        # Correct flickering detections among a maximum of max_age frames
        if self.json_data == {}:
            return
        max_age, from_frame, to_frame = self.getFlickeringValues()
        # self.json_data = correct_flickering(self.json_data, max_age)
        self.json_data = correct_flickering_barrido(
            self.json_data, max_age, from_frame, to_frame
        )
        self.label_flickerin_status.setText("Done")
        self.updateImage()

    def getNearestID(self):
        if self.json_data == {}:
            return
        # Show the nearest id to the bbox clicked
        for det in self.json_data[str(self.frame_pos + 1)]:
            bbox = det["BboxP"]
            if (
                bbox[0] <= self.bbox[0] <= bbox[2]
                and bbox[1] <= self.bbox[1] <= bbox[3]
            ):
                self.text_actId.setText(str(det["id"]))
                self.text_fromFrame.setText(str(self.frame_pos + 1))
                self.clicked_det = det
                break

    def label_clicked(self, rect):
        print(f"Clicked at coordinates: ({rect})")
        if self.pixmap is None:
            return

        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        print(f"Clicked at coordinates: ({x0}, {y0}, {x1}, {y1})")

        # print(f"Label size: {self.label_VideoFrame.width()}, {self.label_VideoFrame.height()}")
        # print(f"Pixmap size: {self.label_VideoFrame.pixmap().width()}, {self.label_VideoFrame.pixmap().height()}")
        # print(f"Image size: {self.label_VideoFrame.pixmap().toImage().width()}, {self.label_VideoFrame.pixmap().toImage().height()}")
        # print(f"Image size: {self.img_width}, {self.img_height}")

        scale_w = self.img_width / self.label_VideoFrame.pixmap().width()
        scale_h = self.img_height / self.label_VideoFrame.pixmap().height()

        # Calculate the pixmap width taking into acount it is centered in the label
        pixmap_x0 = x0 - (
            (self.label_VideoFrame.width() - self.label_VideoFrame.pixmap().width()) / 2
        )
        pixmap_x1 = x1 - (
            (self.label_VideoFrame.width() - self.label_VideoFrame.pixmap().width()) / 2
        )
        pixmap_y0 = y0 - (
            (self.label_VideoFrame.height() - self.label_VideoFrame.pixmap().height())
            / 2
        )
        pixmap_y1 = y1 - (
            (self.label_VideoFrame.height() - self.label_VideoFrame.pixmap().height())
            / 2
        )

        original_x0 = int(pixmap_x0 * scale_w)
        original_x1 = int(pixmap_x1 * scale_w)
        original_y0 = int(pixmap_y0 * scale_h)
        original_y1 = int(pixmap_y1 * scale_h)

        # #Adjust width image coordinate limits
        original_x0 = max(0, min(original_x0, self.img_width))
        original_x1 = max(0, min(original_x1, self.img_width))
        original_y0 = max(0, min(original_y0, self.img_height))
        original_y1 = max(0, min(original_y1, self.img_height))

        print(
            f"Original image clicked at: ({original_x0}, {original_y0}, {original_x1}, {original_y1})"
        )
        self.bbox = [original_x0, original_y0, original_x1, original_y1]

        # Check if the clicked was for bbox creation or for bbox selection
        if original_x0 == original_x1 and original_y0 == original_y1:
            self.getNearestID()

    def convert_text_to_int(self, text_content):
        try:
            result = int(text_content)
            # print(f"Converted integer: {result}")
            return result
        except ValueError:
            print("The text is not a valid integer")
            self.show_error_message("The text is not a valid integer")

    def save_lasts_annotations(self, n=-5):
        # Save the last 5 annotations
        self.json_data_previous.append(deepcopy(self.json_data))
        self.json_data_previous = self.json_data_previous[n:]
        print("Last annotations saved")

    def removeAnnotation(self):
        # Remove the last annotation (Ctrl+z option)
        print("Undo")
        if self.json_data_previous == {}:
            return
        self.json_data = self.json_data_previous.pop()
        self.updateImage()

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_question_message(self, title, message):
        QMessageBox.question(
            self, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )

    def closeEvent(self, event):
        # Release the video capture
        if self.cap is not None:
            self.cap.release()

        if not self.save_status:
            # Ask if the user wants to save the changes
            reply = self.show_question_message(
                "Save", "Do you want to save the changes?"
            )
            if reply == QMessageBox.Yes:
                self.saveResults()

        event.accept()


def main():
    # This line is really IMPORTANT!!!! This is necesary to avoid conlficts when PyQt and opencv running together!
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(
        QtCore.QLibraryInfo.PluginsPath
    )
    app = QApplication(sys.argv)
    player = MainWindow()
    player.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
