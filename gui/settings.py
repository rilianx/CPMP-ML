from copy import deepcopy
import os
import json

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib

from views import SettingsView

class Settings:
    

    file_settings_path = os.path.expanduser("~/.config/cpmp.json")

    default_config = {
       "stack_count"     :  5,
       "stack_height"    :  5,
       "container_count" :  10,
       "sample_size"     :  1000
    }


    def __init__(self):
        s=self

        # Init config map to populate it with file
        self.settings_data = deepcopy(self.default_config)
        
        # Load settings from file
        init_config = False
        if os.path.isfile(s.file_settings_path):
            print("Settings: Found settings file")
            with open(s.file_settings_path, 'r') as file:
                try:
                    self.settings_data = json.load(file)
                except ValueError as _:
                    print(f"Settings: '{s.file_settings_path}' is not valid json")
                    init_config = True
        else: 
            print("Initializing configuration")
            init_config = True

        if init_config:
            s.save_settings_default()
        else: 
            print("Settings: Succesfully loaded settings")

        # Make sure the settings are initialized
        self.view = SettingsView(self.settings_data)



        # # Populate settings dialog
        #
        # self.dialog = Gtk.Dialog()
        # self.dialog.add_buttons(
        #     Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
        #     Gtk.STOCK_OK, Gtk.ResponseType.OK)
        # vbox = self.dialog.get_content_area()
        # vbox.add(self.view)

    # def show_view(self):
    #     self.dialog.show_all()
    #     button = self.dialog.run()
    #     if button == Gtk.ResponseType.OK:
    #         print("Dialog OK pressed: Saving settings")
    #         # Update data with view
    #         self.settings_data = self.view.settings_data
    #         self.save_settings(self.settings_data)
    #     else:
    #         print("SettingsDialog: Dialog Cancel pressed")
    #     self.dialog.hide()
    def save_settings_default(self):
        self.save_settings(self.default_config)

    def save_settings_callback(self, _):
        self.settings_data = self.view.settings_data
        self.save_settings(self.settings_data)

    def save_settings(self, data):
        print(f"Settings: Saving '{self.file_settings_path}'")
        print (f"data: {data}")
        with open(self.file_settings_path, 'w') as output_file:
            json.dump(data, output_file)
        # self.settings_data = data
        # # Refresh settings on dialog exit
        # self.S = self.settings_data['stack_count']
        # self.H = self.settings_data['stack_height']
        # self.N = self.settings_data['container_count']



