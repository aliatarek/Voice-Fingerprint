# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
import Interface
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget
import sys
import test5

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi("Task5.ui",self)
        Interface.initConnectors(self)
        self.show()
        test5.train_model(self)

            
app=QApplication(sys.argv)
UIWindow= UI()
app.exec_()

    