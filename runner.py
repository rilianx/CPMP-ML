import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import asyncio
class Runner:
    def __init__(self, buf):
        self.buf = buf
        self.stdout_q = asyncio.Queue()
        self.stderr_q = asyncio.Queue()

    async def run_subprocess(self, cmd):
        for arg in cmd: print(arg, end = " ", file = self.buf)
        print(file=self.buf)

        process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
                )

        asyncio.ensure_future(self.read_output(process.stdout, self.stdout_q))
        asyncio.ensure_future(self.read_output(process.stderr, self.stderr_q))

        asyncio.ensure_future(self.write_to_buffer(self.stdout_q))
        asyncio.ensure_future(self.write_to_buffer(self.stderr_q))

        retcode = await process.wait()
        # stdout, stderr = await process.communicate()

        dialog = Gtk.MessageDialog(
            buttons=Gtk.ButtonsType.OK,
            text=f"Process finished. Exit status : {retcode}"
        )
        dialog.run()
        dialog.destroy()

        self.stdout_q.put_nowait(None)
        self.stderr_q.put_nowait(None)


        # Append the output to the buffer asynchronously
        # asyncio.ensure_future(self.append_output(stdout, stderr))

    async def append_output(self, stdout, stderr):
        # Decode the stdout and stderr and append them to the buffer
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()
        end_iter = self.buf.get_end_iter()
        self.buf.insert(end_iter, stdout_str + stderr_str)

    async def read_output(self, stream, queue):
        while True:
            line = await stream.readline()
            if not line: break
            await queue.put(line)
    async def write_to_buffer(self, queue):
        while True:
            line = await queue.get()
            if not line: break
            self.buf.write(line.decode())
            self.buf.flush()

