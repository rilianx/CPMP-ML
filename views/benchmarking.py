import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from cpmp_ml import validate_model

with open('views/benchmarking.glade', 'r') as file:
    xml = file.read()

@Gtk.Template(string=xml)
class Benchmarking(Gtk.Box):


    __gtype_name__ = "benchmarking"

    input_model_btn = Gtk.Template.Child()
    stack_count = Gtk.Template.Child()

    model = None
    S=5;
    H=5;
    N=15;
    ss=1000


    @Gtk.Template.Callback()
    def openSettings(self, *args):
        print("hello")


    @Gtk.Template.Callback()
    def input_model_clicked(self, *args):
        print("Select your input model")
        f=Benchmarking.get_file()
        if f:
            self.input_model_btn.set_label(f.split("/")[-1])
            self.model = create_model(self.S, self.H)
        else: print("Error file:", f)
        pass

    @Gtk.Template.Callback()
    def validate_model_clicked(self, *args):
        if  self.model == None:
            Gtk.MessageDialog(parent = None, flags = 0,
                              text = "Error: You must load a model first").run()
        else:
            x_test, y_test = generate_data(sample_size=self.ss, S=self.S, H=self.H, N=self.N)
            test_loss, test_acc = self.model.evaluate(np.array(x_test), np.array(y_test))
            print("test_loss: ", test_loss)
            print("test_acc: ", test_acc)
        pass
        


    @Gtk.Template.Callback()
    def stack_count_value_changed(self, spin_button, *args):
        self.S = spin_button.get_value_as_int()
        print("New stack count:", self.S)
        return 0

    @Gtk.Template.Callback()
    def stack_height_value_changed(self, spin_button, *args):
        self.H = spin_button.get_value_as_int()
        print("New stack height:", self.H)
        return 0

    @Gtk.Template.Callback()
    def container_count_value_changed(self, spin_button, *args):
        self.N = spin_button.get_value_as_int()
        print("New container count:", self.N)
        return 0

    @Gtk.Template.Callback()
    def sample_size_input_changed(self, spin_button, *args):
        self.ss = spin_button.get_value_as_int()
        print("New random sample size:", self.ss)
        return 0





    def get_file():
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
