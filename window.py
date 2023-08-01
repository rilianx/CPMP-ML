#!/usr/bin/env python
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import asyncio, gbulb
gbulb.install(gtk=True)

from views import GenerateDataView, TrainModelView

from views.builder import Builder
from gui import Settings
from redirector import TextBufferOutput
from runner import Runner

class App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="rilianx.cpmp")
        self.model        = None

        builder           = Builder()
        builder.add_from_file('./views/window.glade')

        # Get glade objects

        self.window    = builder.get_casted(Gtk.Window, 'root-window')
        self.menu_bar  = builder.get_casted(Gtk.MenuBar, 'menu-bar')
        self.paned     = builder.get_casted(Gtk.Paned, 'paned')

        self.settings = Settings()

        self.stack = Gtk.Stack(
                transition_type=Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
                )

        self.log_view_buffer = Gtk.TextBuffer()

        self.log_view = Gtk.TextView(
                editable=False,
                buffer = self.log_view_buffer,
                wrap_mode=Gtk.WrapMode.WORD,
                cursor_visible=True,
                monospace=True
                )
        self.scrolled_log_view = Gtk.ScrolledWindow()
        self.scrolled_log_view.set_policy(
                Gtk.PolicyType.AUTOMATIC,
                Gtk.PolicyType.AUTOMATIC
                )
        self.scrolled_log_view.add(self.log_view)

        # Black magic. Props to GPT
        self.buf = TextBufferOutput(self.log_view_buffer)
        self.runner = Runner(self.buf)

        self.generate_data = GenerateDataView(self.settings.view, self.runner)
        self.train_model = TrainModelView(self.buf, self.settings, self.model, self.runner)
        # self.settings.dialog.get_content_area().remove(self.settings.view)
        self.paned.add1(self.settings.view)

        vpane = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        vpane.add1(self.stack)
        vpane.add2(self.scrolled_log_view)


        self.paned.add2(vpane)

        self.main = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.main.pack_start(self.generate_data, False, False, 10)
        self.main.pack_start(self.train_model, False, False, 10)

        self.settings.view.show_all()
        self.settings.view.save_btn.connect('clicked', self.settings.save_settings_callback)
        self.stack.add_named(self.main, "main")

        #

        # Bind objects
        # builder.get_casted(Gtk.Button, 'open-model-menu') \
            # .connect('activate', self.benchmarking.input_model_clicked)
        #
        # builder.get_casted(Gtk.MenuItem, 'settings-menu') \
        #     .connect('activate', self.open_settings)
        #
        # builder.get_casted(Gtk.MenuItem, 'generate-data-menu') \
        #         .connect('activate', self.open_generate_data)
        #
        # builder.get_casted(Gtk.MenuItem, 'benchmarking-menu') \
        #         .connect('activate', self.open_benchmarking)


        # Insert main view
        # self.stack.add_named(self.generate_data, name = 'generate-data')
        # self.stack.add_named(self.benchmarking, name = 'benchmarking')

        # Not working xd
        # self.stack.set_visible_child_name('benchmarking')

        # Add Settings view to generate-data


    def on_quit(self, _):
        self.settings.save_settings_callback(_)
        Gtk.main_quit()

    def do_activate(self):
        self.window.connect("destroy", self.on_quit)
        self.window.show_all()
        Gtk.main()

    # Redirection to member class
    def open_generate_data(self, _):
        self.stack.set_visible_child_name('generate-data')

    def open_benchmarking(self, _):
        self.stack.set_visible_child_name('benchmarking')

    # def open_settings(self, _):
    #     self.settings.show_view()

    # def open_model(self, _):
    #     self.benchmarking.input_model_clicked(_)

    # def show_view(self):
    #     self.dialog.show_all()
    #     button = self.dialog.run()
    #     if button == Gtk.ResponseType.OK:
    #         print("Dialog OK pressed: Saving settings")
    #         self.settings_data['stack_count'] = self.view.S
    #         self.settings_data['stack_height'] = self.view.H
    #         self.settings_data['container_count'] = self.view.N
    #         self.save_settings(self.settings_data)
    #     else:
    #         print("SettingsDialog: Dialog Cancel pressed")
    #     self.dialog.hide()
    #     print("Settings")
        

        
if __name__ == "__main__":
    print(f'GTK Version: {Gtk.MAJOR_VERSION}.' +
          f'{Gtk.MINOR_VERSION}.{Gtk.MICRO_VERSION}')
    app = App()
    loop = asyncio.get_event_loop()
    loop.run_forever(application=app)
    # exit_status = app.run(sys.argv)
    # sys.exit(exit_status)

