import io
class TextBufferOutput(io.TextIOBase):
    def __init__(self, text_buffer):
        super().__init__()
        self.text_buffer = text_buffer

    def write(self, s):
        end_iter = self.text_buffer.get_end_iter()
        self.text_buffer.insert(end_iter, s)
        return len(s)

    def flush(self):
        pass
