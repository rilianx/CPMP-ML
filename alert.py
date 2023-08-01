import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject

class Dialog:
    def __init__(self, string):
        dialog = Gtk.MessageDialog(
            buttons=Gtk.ButtonsType.OK,
            text=string
        )
        dialog.run()
        dialog.destroy()

