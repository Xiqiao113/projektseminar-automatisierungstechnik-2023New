from PyQt5.Qt import *
from Chooseui import *
from Aufgabe1Ui import *
from Aufgabe2Ui import *
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.Aufgabe1_btn.clicked.connect(self.login_in1)
        self.ui.Aufgabe2_btn.clicked.connect(self.login_in2)
        self.show()


    def login_in1(self):
        self.win = Aufgabe1Window()
        self.close()

    def login_in2(self):
        self.win = Aufgabe2Window()
        self.close()



class Aufgabe1Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Aufgabe1_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.Aufgabe1_back_btn.clicked.connect(self.back)
        self.ui.K_lineEdit.textChanged.connect(self.enable_submit_btn1)
        self.ui.S_lineEdit.textChanged.connect(self.enable_submit_btn1)
        self.ui.sub_1.clicked.connect(self.get_infos)
        self.show()

    def back(self):
        self.win = LoginWindow()
        self.close()

    def get_infos(self):
        Karte_Nr = self.ui.K_lineEdit.text()
        Szen_Nr = self.ui.S_lineEdit.text()


        print("the chosen Szenario is map:" , Karte_Nr , 'Senario:' ,Szen_Nr)

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
    def __init__(self):
        super().__init__()
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
        self.show()

    def back(self):
        self.win = LoginWindow()
        self.close()

    def submit(self):
        R_x = self.ui.Rover_x.text()
        R_y = self.ui.Rover_y.text()

        P_x = self.ui.Probe_x.text()
        P_y = self.ui.Probe_y.text()

        self.x, self.y = get_coordinates(R_x, R_y, P_x, P_y)

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
            if  0 <= R_x <= 2800 and 0 <= R_y <= 2800 and 0 <= P_x <= 2800 and 0 <= P_y <= 2800:
                self.ui.sub_2.setEnabled(True)
            else:
                self.ui.sub_2.setEnabled(False)
        except ValueError:
            self.ui.sub_2.setEnabled(False)





if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) #
    app = QApplication(sys.argv)

    def get_coordinates(x1_str, y1_str, x2_str, y2_str):
        x1 = float(x1_str)
        y1 = float(y1_str)
        x2 = float(x2_str)
        y2 = float(y2_str)

        Rover_coord = [x1, y1]
        Probe_coord = [x2, y2]

        print("Rover:  =", Rover_coord)
        print("Probe:  =", Probe_coord)

        return  Rover_coord, Probe_coord

    win = LoginWindow()
    sys.exit(app.exec_())



