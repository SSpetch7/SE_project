from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.label import Label
import cv2 as cv

class MainMenu(Screen):
    def loadVideo(self):
        self.capture = cv.VideoCapture('Videos/jump.mp4')
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            self.frame = frame
            
            cv.imshow('Squat', self.frame)

            if cv.waitKey(20) & 0xFF == ord('d'):
                break

class Dungeon(Screen):
    pass


class RisingKnee(Screen):
    pass


class RisingKneePlay(Screen):
    pass


class Setting(Screen):
    pass


class WindowManager(ScreenManager):
    pass

kv = Builder.load_file('thelab.kv')

class TheLabApp(App):
    
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        return kv

if __name__ == '__main__':
    TheLabApp().run()