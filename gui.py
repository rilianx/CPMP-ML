#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib
import sys
from cpmp_ml import create_model, generate_data
import numpy as np
from views.benchmarking import Benchmarking



class Window(Gtk.Window):
    def __init__(self):
        super().__init__(title = "Benchmarking")
        self.rootView = Gtk.Box()
        self.rootView.set_orientation(Gtk.Orientation.VERTICAL)
        self.contentView = Benchmarking()
        self.add(self.rootView)
        self.rootView.pack_start(self.contentView, True, True, 0)



print(f'Using GTK {Gtk.MAJOR_VERSION}.{Gtk.MINOR_VERSION}.{Gtk.MICRO_VERSION}')


class MyApplication(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.MyGtkApplication")
        GLib.set_application_name('My Gtk Application')

    def do_activate(self):
        win = Window()
        win.connect("destroy", Gtk.main_quit)
        win.show_all()
        Gtk.main()
        



app = MyApplication()
exit_status = app.run(sys.argv)
sys.exit(exit_status)
