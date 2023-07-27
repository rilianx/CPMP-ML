import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from cpmp_ml import validate_model, create_model
from views.settings import Settings

class Benchmarking():
    pass


with open('views/benchmarking.glade', 'r') as file:
    xml = file.read()

@Gtk.Template(string=xml) # type: ignore
class BenchmarkingView(Gtk.Box):

    def __init__(self):
        super().__init__()
        self.settings = Settings()

    __gtype_name__ = "benchmarking"

    input_model_btn: Gtk.Button = Gtk.Template.Child() # type: ignore

    model = None


    @Gtk.Template.Callback() # type: ignore
    def open_settings_view(self, *args):
        self.settings.show_view()


    @Gtk.Template.Callback()
    def input_model_clicked(self, *args):
        print("Select your input model")
        f=BenchmarkingView.get_file_dialog()
        if f:
            self.input_model_btn.set_label(f.split("/")[-1])
            self.model = create_model(self.settings.S, self.settings.H)
        else: print("Error file:", f)
        pass

    @Gtk.Template.Callback()
    def validate_model_clicked(self, *args):
        print("validate_model_clicked")
        if  self.model == None:
            Gtk.MessageDialog(parent = self, flags = 0,
                              text = "Error: You must load a model first").run()
        try: 
            validate_model(self.model, 
                        self.settings.S,
                        self.settings.H,
                        self.settings.N,
                        self.ss)
        except RuntimeError as err:
            # TODO:
            # invalid_model_dialog()
            pass
        # else:
        #     x_test, y_test = generate_data(sample_size=self.ss, S=self.S, H=self.H, N=self.N)
        #     test_loss, test_acc = self.model.evaluate(np.array(x_test), np.array(y_test))
        #     print("test_loss: ", test_loss)
        #     print("test_acc: ", test_acc)
        pass
        



    @Gtk.Template.Callback() # type: ignore
    def sample_size_input_changed(self, spin_button, *args):
        self.ss = spin_button.get_value_as_int()
        print("New random sample size:", self.ss)
        return 0





    def get_file_dialog():
        dialog = Gtk.FileChooserDialog(
                title="Select a file",
                parent=None,
                action=Gtk.FileChooserAction.OPEN,
                buttons=(
                    Gtk.STOCK_CANCEL,
                    Gtk.ResponseType.CANCEL,
                    Gtk.STOCK_OPEN,
                    Gtk.ResponseType.OK
                )
            )

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            selected_file = dialog.get_filename()
            dialog.destroy()
            print("Selected file:", selected_file)
            return selected_file
        return None
