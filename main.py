from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.label import Label

class MainMenu(Screen):
    pass


class Dungeon(Screen):
    pass


class RisingKnee(Screen):
    pass


class RisingKneePlay(Screen):
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