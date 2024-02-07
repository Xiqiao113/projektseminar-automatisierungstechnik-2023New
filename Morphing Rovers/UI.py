from PyQt5.Qt import *
from Chooseui import *
from Aufgabe1Ui import *
from Aufgabe2Ui import *
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import numpy as np

global_x_value = None


class LoginWindow(QMainWindow):
    xValueUpdated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.x = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.Aufgabe1_btn.clicked.connect(self.getAufgabe1)
        self.ui.Aufgabe1_btn.clicked.connect(self.login_in1)
        self.ui.Aufgabe2_btn.clicked.connect(self.getAufgabe2)
        self.ui.Aufgabe2_btn.clicked.connect(self.login_in2)
        self.show()

    def login_in1(self):
        self.win = Aufgabe1Window(self)  # Pass self as parent
        self.hide()

    def getAufgabe1(self):
        self.x = 1
        self.xValueUpdated.emit(self.x)

    def login_in2(self):
        self.win = Aufgabe2Window(self)
        self.hide()

    def getAufgabe2(self):
        self.x = 2
        self.xValueUpdated.emit(self.x)


class Aufgabe1Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_Aufgabe1_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.Aufgabe1_back_btn.clicked.connect(self.back)
        self.ui.K_lineEdit.textChanged.connect(self.enable_submit_btn1)
        self.ui.S_lineEdit.textChanged.connect(self.enable_submit_btn1)
        self.ui.sub_1.clicked.connect(self.get_infos)
        self.parent = parent
        self.show()

    def back(self):
        if self.parent:
            self.parent.show()  # Use the saved reference to redisplay the LoginWindow.
        self.close()

    def get_infos(self):
        Karte_Nr = self.ui.K_lineEdit.text()
        Szen_Nr = self.ui.S_lineEdit.text()
        try:
            Karte_Nr_int = int(Karte_Nr)
            Szen_Nr_int = int(Szen_Nr)

            with open('user_input.txt', 'a') as file:
                file.write(f"\t{Karte_Nr_int}\t{Szen_Nr_int}\n")  # Formatting strings with newlines
        except ValueError:

            print("输入的值不是有效的整数")


    def enable_submit_btn1(self):
        Karte_Nr = self.ui.K_lineEdit.text()
        Szen_Nr = self.ui.S_lineEdit.text()

        try:
            Karte_Nr = int(Karte_Nr)
            Szen_Nr = int(Szen_Nr)
            if 1 <= Karte_Nr <= 6 and 1 <= Szen_Nr <= 5:
                self.ui.sub_1.setEnabled(True)
            else:
                self.ui.sub_1.setEnabled(False)
        except ValueError:
            self.ui.sub_1.setEnabled(False)


class Aufgabe2Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_Aufgabe2_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.Aufgabe2_back_btn.clicked.connect(self.back)
        self.ui.Rover_x.textChanged.connect(self.enable_submit_btn2)
        self.ui.Rover_y.textChanged.connect(self.enable_submit_btn2)
        self.ui.Probe_x.textChanged.connect(self.enable_submit_btn2)
        self.ui.Probe_y.textChanged.connect(self.enable_submit_btn2)
        self.ui.sub_2.clicked.connect(self.submit)
        self.parent = parent
        self.show()

    def back(self):
        if self.parent:
            self.parent.show()  # Use the saved reference to redisplay the LoginWindow.
        self.close()

    def submit(self):
        R_x = self.ui.Rover_x.text()
        R_y = self.ui.Rover_y.text()

        P_x = self.ui.Probe_x.text()
        P_y = self.ui.Probe_y.text()

        try:
            R_x_float = float(R_x)
            R_y_float = float(R_y)
            P_x_float = float(P_x)
            P_y_float = float(P_y)
            # write to a file
            with open('user_input.txt', 'a') as file:  # Using the append mode
                file.write(f"\t{R_x_float}\t{R_y_float}\t{P_x_float}\t{P_y_float}\n")  # Use \t for Tab Separation
        except ValueError:
            print("not valuable float")

    def enable_submit_btn2(self):
        R_x = self.ui.Rover_x.text()
        R_y = self.ui.Rover_y.text()

        P_x = self.ui.Probe_x.text()
        P_y = self.ui.Probe_y.text()

        try:
            R_x = float(R_x)
            R_y = float(R_y)
            P_x = float(P_x)
            P_y = float(P_y)
            if 0 <= R_x <= 2800 and 0 <= R_y <= 2800 and 0 <= P_x <= 2800 and 0 <= P_y <= 2800:
                self.ui.sub_2.setEnabled(True)
            else:
                self.ui.sub_2.setEnabled(False)
        except ValueError:
            self.ui.sub_2.setEnabled(False)

# from texttext import collect_data
def handleXValueUpdated(value):
    # print("new x:", value)
    with open('user_input.txt', 'w') as file:
        file.write(str(value))


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  #
    app = QApplication(sys.argv)
    win = LoginWindow()
    win.xValueUpdated.connect(handleXValueUpdated)  # Connecting Signals to Slot Functions
    win.show()
    result = app.exec_()

    sys.exit(result)


