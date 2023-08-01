import asyncio
import io
import logging
import sys
from typing import cast

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject

from cpmp_ml import load_model, validate_model, create_model
from alert import Dialog


with open('views/train_model.glade', 'r') as file:
    xml = file.read()

@Gtk.Template(string=xml) # type: ignore
class TrainModelView(Gtk.Box):


    __gtype_name__ = "train-model"
    input_data_name = previous_weights_name = None
    output_name_entry = cast(Gtk.Entry, Gtk.Template.Child())

    # __gsignals__ = {
    #         "open-model-btn": ((GObject.SignalFlags.RUN_FIRST, None, ()))
    # }

    def __init__(self, buf, settings, model, process_runner):
        super().__init__()
        self.buf = buf
        self.settings = settings
        self.model = model
        self.runner = process_runner
        # redirect_to_text_buffer(self.logger_buffer)


    @Gtk.Template.Callback() #type: ignore
    def previous_weights_btn_cb(self, btn):
        self.previous_weights_name = btn.get_filename()

    @Gtk.Template.Callback() #type: ignore
    def input_data_btn_cb(self, btn):
        self.input_data_name = btn.get_filename()

    @Gtk.Template.Callback() #type: ignore
    def train_model_btn_cb(self, btn):
        print ("Train model")
        if self.input_data_name == None:
            Dialog("Error: Choose an input data file")
            return
            
        cmd = ["./__main__.py", "-tm",
            "-i", self.input_data_name,
            "-S", f"{self.settings.view.get_s()}",
            "-H", f"{self.settings.view.get_h()}",
            "-N", f"{self.settings.view.get_n()}",
            "-ss", f"{self.settings.view.get_ss()}",
            "-e", f"{10}",
            "-o", self.output_name_entry.get_text()
        ]
        for arg in cmd: print(arg, end = " ", file = self.buf)

        asyncio.ensure_future(self.runner.run_subprocess(cmd))

        return 0
        pass

    # def get_file_dialog():
    #     dialog = Gtk.FileChooserDialog(
    #             title="Select a file",
    #             parent=None,
    #             action=Gtk.FileChooserAction.OPEN,
    #             buttons=(
    #                 Gtk.STOCK_CANCEL,
    #                 Gtk.ResponseType.CANCEL,
    #                 Gtk.STOCK_OPEN,
    #                 Gtk.ResponseType.OK
    #             )
    #         )
    #     selected_file = None
    #     response = dialog.run()
    #     if response == Gtk.ResponseType.OK:
    #         selected_file = dialog.get_filename()
    #         print("Selected file:", selected_file)
    #     dialog.hide()
    #     return selected_file

# def redirect_to_text_buffer(text_buffer):
#     class GtkTextBufferHandler(logging.Handler):
#         def __init__(self, target_text_buffer):
#             super().__init__()
#             self.target_text_buffer = target_text_buffer
#
#         def emit(self, record):
#             # Update the Gtk.TextBuffer with the log message
#             msg = self.format(record) + "\n"
#             end_iter = self.target_text_buffer.get_end_iter()
#             self.target_text_buffer.insert(end_iter, msg)
#
#     class RedirectOutput(io.TextIOBase):
#         def __init__(self, target_text_buffer):
#             self.target_text_buffer = target_text_buffer
#
#         def write(self, s):
#             # Update the Gtk.TextBuffer with the captured output
#             end_iter = self.target_text_buffer.get_end_iter()
#             self.target_text_buffer.insert(end_iter, s)
#
#     # Create a custom logger for TensorFlow
#     tensorflow_logger = logging.getLogger("tensorflow")
#     tensorflow_logger.setLevel(logging.DEBUG)
#     sys.stderr = RedirectOutput(text_buffer)
#
#     # Redirect TensorFlow log messages to the custom logger
#     handler = GtkTextBufferHandler(text_buffer)
#     tensorflow_logger.addHandler(handler)
