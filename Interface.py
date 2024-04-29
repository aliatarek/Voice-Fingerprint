from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton,QLCDNumber
from pyqtgraph import PlotWidget
from mplwidget import MplWidget
import pyqtgraph
import test5
import HandleStateChange

#from classes import channelLine




def initConnectors(self):
    
   
    
    self.Spectrogram=self.findChild(MplWidget , "Spectrogram") 
    #self.inputspectogram.setBackground('w')

    self.Mic=self.findChild(QtWidgets.QPushButton,"Mic")
   
    self.Mic.clicked.connect(lambda:test5.start_recording(self))
    
    self.Mode1=self.findChild(QtWidgets.QRadioButton,"Mode1")
    self.Mode1.toggled.connect(lambda:test5.whichmode(self,1))

    self.Mode2=self.findChild(QtWidgets.QRadioButton,"Mode2")
    self.Mode2.toggled.connect(lambda:test5.whichmode(self,2))

    
    self.table1=self.findChild(QtWidgets.QTableWidget,"table1")
    
    self.table2=self.findChild(QtWidgets.QTableWidget,"table2")
    
    self.select_users=[0]*12
    
    self.Amr=self.findChild(QtWidgets.QCheckBox,"Amr")
    self.Amr.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,1))

    self.Alia=self.findChild(QtWidgets.QCheckBox,"Alia")
    self.Alia.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,2))

    self.Hamza=self.findChild(QtWidgets.QCheckBox,"Hamza")
    self.Hamza.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,3))

    self.Mahmoud=self.findChild(QtWidgets.QCheckBox,"Mahmoud")
    self.Mahmoud.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,4))
        
    self.Awad=self.findChild(QtWidgets.QCheckBox,"Awad")
    self.Awad.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,5))
       
    self.Tarek=self.findChild(QtWidgets.QCheckBox,"Tarek")
    self.Tarek.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,6))
    
    self.Afify=self.findChild(QtWidgets.QCheckBox,"Afify")
    self.Afify.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,7))
        
    self.Mohamed=self.findChild(QtWidgets.QCheckBox,"Mohamed")
    self.Mohamed.stateChanged.connect(lambda:HandleStateChange.handleStateChanged(self,8))
    
    
    self.users=[]
    self.users.append(self.Amr)
    self.users.append(self.Alia)
    self.users.append(self.Hamza)
    self.users.append(self.Mahmoud)
    self.users.append(self.Awad)
    self.users.append(self.Tarek)
    self.users.append(self.Afify)
    self.users.append(self.Mohamed)
    
    
    
    self.recordingStatus=self.findChild(QtWidgets.QLabel,"recordingStatus")
    self.AccessStatus=self.findChild(QtWidgets.QLabel,"AccessStatus")


