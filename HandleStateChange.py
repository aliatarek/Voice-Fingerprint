from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton,QLCDNumber
from pyqtgraph import PlotWidget
from mplwidget import MplWidget
import pyqtgraph
import test5





def handleStateChanged(self,caller):
    if caller==1:
        if self.Amr.isChecked():
            for i in range(3):
                self.select_users[i]=1
                self.index=1
        else:
            for i in range(3):
                if self.select_users[i]:
                 self.select_users[i]*0
    if caller==2:
        if self.Alia.isChecked():
            i=3
            for i in range(6):
                self.select_users[i]=1
                self.index=3
        else:
            i=3
            for i in range(6):
                if self.select_users[i]:
                 self.select_users[i]*0
    if caller==3:
        if self.Hamza.isChecked():
            i=6
            for i in range(9):
                self.select_users[i]=1
                self.index=6
        else:
            i=6
            for i in range(9):
                if self.select_users[i]:
                 self.select_users[i]*0
                 self.index=9
            
    if caller==4:
        if self.Mahmoud.isChecked():
            i=9
            for i in range(12):
                self.select_users[i]=1
                self.index=12
        else:
            i=9
            for i in range(12):
                if self.select_users[i]:
                 self.select_users[i]*0

    # if caller==5:
    #     if self.Awad.isChecked():
    #         i=12
    #         for i in range(5):
    #             self.select_users[i]=1
    #             self.index=15
    #     else:
    #         i=4
    #         for i in range(5):
    #             if self.select_users[i]:
    #              self.select_users[i]*0
    # if caller==6:
    #     if self.Tarek.isChecked():
    #         i=5
    #         for i in range(6):
    #             self.select_users[i]=1
    #             self.index=18
    #     else:
    #         i=5
    #         for i in range(6):
    #             if self.select_users[i]:
    #              self.select_users[i]*0
    # if caller==7:
    #     if self.Afify.isChecked():
    #         i=6
    #         for i in range(7):
    #             self.select_users[i]=1
    #             self.index=21
    #     else:
    #         i=6
    #         for i in range(7):
    #             if self.select_users[i]:
    #              self.select_users[i]*0
    # if caller==8:
    #     if self.Mohamed.isChecked():
    #         i=7
    #         for i in range(8):
    #             self.select_users[i]=1
    #             self.index=24
    #     else:
    #         i=7
    #         for i in range(8):
    #             if self.select_users[i]:
    #              self.select_users[i]*0