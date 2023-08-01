import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from typing import cast
from views.builder import Builder
import asyncio

filename='views/generate_data.glade'
class GenerateDataView(Gtk.Box):

    __gtype_name__ = "generate-data"
    btn = cast(Gtk.Button, Gtk.Template.Child())

    def do_generate_data(self, _: Gtk.Button):
        cmd = ["./__main__.py", "-gd",
            "-S", f"{self.settings_view.get_s()}",
            "-H", f"{self.settings_view.get_h()}",
            "-N", f"{self.settings_view.get_n()}",
            "-ss", f"{self.settings_view.get_ss()}",
            "-o", f"{self.output_name_entry.get_text()}"
        ]

        asyncio.ensure_future(self.runner.run_subprocess(cmd))

        return 0


    def __init__(self, settingsView, runner):
        super().__init__()
        self.settings_view = settingsView
        self.runner = runner


        builder = Builder()
        builder.add_from_file(filename)

        self.root_box = builder.get_casted(Gtk.Box, 'generate-data')
        self.btn = builder.get_casted(Gtk.Button, 'generate-data-btn')
        self.output_name_entry = builder.get_casted(Gtk.Entry, 'output-name-entry')

        # self.settings = settings
        self.btn.connect('clicked', self.do_generate_data)
        self.add(self.root_box)


