import gi
import json
import os.path
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


class Settings:
    default_config = {
       "stack_count"     :  5,
       "stack_height"    :  5,
       "container_count" : 10,
    }

    file_settings_path = os.path.expanduser("~/.config/cpmp.json")


    def __init__(self):
        s=self
        self.view = SettingsView()

        # Load settings from file
        init_config = False
        if os.path.isfile(s.file_settings_path):
            print("Settings: Found settings file")
            with open(s.file_settings_path, 'r') as file:
                try:
                    self.settings_data = json.load(file)
                except ValueError as e:
                    print(f"Settings: '{s.file_settings_path}' is not valid json")
                    init_config = True
        else: init_config = True
        if init_config: s.save_settings(self.default_config)
        else: 
            print("Settings: Succesfully loaded settings")

        # Make sure the settings are initialized
        self.S = self.settings_data['stack_count']
        self.H = self.settings_data['stack_height']
        self.N = self.settings_data['container_count']

        # Populate spinners with settings

        self.view.stack_count_spinner.set_value(s.settings_data['stack_count'])
        self.view.stack_height_spinner.set_value(s.settings_data['stack_height'])
        self.view.container_count_spinner.set_value(s.settings_data['container_count'])

        # Populate settings dialog

        self.dialog = Gtk.Dialog()
        self.dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK)
        vbox = self.dialog.get_content_area()
        vbox.add(self.view)

    def show_view(self):
        self.dialog.show_all()
        button = self.dialog.run()
        if button == Gtk.ResponseType.OK:
            print("Dialog OK pressed: Saving settings")
            self.settings_data['stack_count'] = self.view.S
            self.settings_data['stack_height'] = self.view.H
            self.settings_data['container_count'] = self.view.N
            self.save_settings(self.settings_data)
        else:
            print("SettingsDialog: Dialog Cancel pressed")
        self.dialog.hide()

    def save_settings(self, data):
        print(f"Settings: Saving '{self.file_settings_path}'")
        with open(self.file_settings_path, 'w') as output_file:
            json.dump(data, output_file)
        self.settings_data = data
        # Refresh settings on dialog exit
        self.S = self.settings_data['stack_count']
        self.H = self.settings_data['stack_height']
        self.N = self.settings_data['container_count']



with open('views/settings.glade', 'r') as file:
    xml = file.read()

@Gtk.Template(string=xml)
class SettingsView(Gtk.Box):
    # TODO: Should recv as __init__ parameters
    S = H = 5
    N = 10

    __gtype_name__ = "settings-view"
    stack_count_spinner = Gtk.Template.Child()
    stack_height_spinner = Gtk.Template.Child()
    container_count_spinner = Gtk.Template.Child()

    @Gtk.Template.Callback()
    def stack_count_value_changed(self, stack_count):
        self.S = stack_count.get_value_as_int()
        print("SettingsView: New stack count:", self.S)
        return 0

    @Gtk.Template.Callback()
    def stack_height_value_changed(self, stack_height):
        self.H = stack_height.get_value_as_int()
        print("SettingsView: New stack height:", self.H)
        return 0

    @Gtk.Template.Callback()
    def container_count_value_changed(self, container_count):
        self.N = container_count.get_value_as_int()
        print("SettingsView: New container count:", self.N)
        return 0
