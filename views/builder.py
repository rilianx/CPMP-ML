import gi
import typing
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

class Builder(Gtk.Builder):
    def get_casted(self, typename, name):
        obj = self.get_object(name)
        if obj == None:
            raise RuntimeError("Builder: NoneType object")
        return typing.cast(typename, obj)
