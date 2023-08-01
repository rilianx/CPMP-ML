import gi
import json
import os.path
from typing import cast
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


with open('views/settings.glade', 'r') as file:
    xml = file.read()

@Gtk.Template(string=xml) #type: ignore
class SettingsView(Gtk.Box):

    __gtype_name__ = "settings-view"

    stack_count_spinner     = cast(Gtk.SpinButton, Gtk.Template.Child())
    stack_height_spinner    = cast(Gtk.SpinButton, Gtk.Template.Child())
    container_count_spinner = cast(Gtk.SpinButton, Gtk.Template.Child())
    sample_size_spinner     = cast(Gtk.SpinButton, Gtk.Template.Child())

    def set_s(self, s):
        self.settings_data['stack_count'] = s

    def get_s(self):
        return self.settings_data['stack_count']

    def set_h(self, h):
        self.settings_data['stack_height'] = h

    def get_h(self):
        return self.settings_data['stack_height']

    def set_n(self, n):
        self.settings_data['container_count'] = n

    def get_n(self):
        return self.settings_data['container_count']

    def set_ss(self, ss):
        self.settings_data['sample_size'] = ss

    def get_ss(self):
        return self.settings_data['sample_size']


    save_btn = cast(Gtk.Button, Gtk.Template.Child())

    def __init__(self, settings_data):
        super().__init__()
        self.settings_data = settings_data

        spinner_getter =  [
                (self.stack_count_spinner,     self.get_s),
                (self.stack_height_spinner,    self.get_h),
                (self.container_count_spinner, self.get_n),
                (self.sample_size_spinner,     self.get_ss),
        ] 
        # print("Loaded", self.settings_data)
        for spinner, getter in spinner_getter:
            spinner.set_value(getter())



    @Gtk.Template.Callback() #type: ignore
    def stack_count_value_changed(self, stack_count):
        self.set_s(stack_count.get_value_as_int())
        # print("SettingsView: New stack count:", self.get_s())
        return 0

    @Gtk.Template.Callback() #type: ignore
    def stack_height_value_changed(self, stack_height):
        self.set_h(stack_height.get_value_as_int())
        # print("SettingsView: New stack height:", self.get_h())
        return 0

    @Gtk.Template.Callback() #type: ignore
    def container_count_value_changed(self, container_count):
        self.set_n(container_count.get_value_as_int())
        # print("SettingsView: New container count:", self.get_n())
        # print (self.settings_data)
        return 0

    @Gtk.Template.Callback() #type: ignore
    def sample_size_value_changed(self, container_count):
        self.set_ss(container_count.get_value_as_int())
        # print("SettingsView: New sample_size:", self.get_ss())
        # print (self.settings_data)
        return 0
