# Xiqiao Zhang
# 2024/2/1
from UI import *

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) #
app = QApplication(sys.argv)
win = LoginWindow()

win.xValueUpdated.connect(handleXValueUpdated)

win.show()
result = app.exec_()

sys.exit(result)


# def readXValueFromFile():
#     try:
#         with open('infos.txt', 'r') as file:
#             value_str = file.read().strip()
#             value = int(value_str)
#             return value
#     except FileNotFoundError:
#         print("File not found.")
#         return None
#     except ValueError:
#         print("Could not convert data to an integer.")
#         return None

# x_value = readXValueFromFile()
# if x_value is not None:
#     print("Read x value from file:", x_value)
# else:
#     print("Failed to read x value from file.")