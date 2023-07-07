#!/usr/bin/env python
# -*- coding: utf-8 -*-



from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
import glfw
import imgui
import sys


def callback(window, width, height):
    p = glfw.get_window_user_pointer(window)
    print(f"{width}x{height}")
    gl.glViewport(0, 0, width, height)
    p.impl.resize_callback(window, width, height)
    p.width = width
    p.height = height
    # p.impl.io.display_size.x = width
    # p.impl.io.display_size.y = height
class GUI:
    window = None


    def __init__(self):
        s=self
        imgui.create_context()
        s.window = self.impl_glfw_init()
        s.impl = GlfwRenderer(s.window)
        s.width, s.height = 1, 1
    def __del__(self):
        s=self
        s.impl.shutdown()
        glfw.terminate()


    def loop(self):
        s=self
        while not glfw.window_should_close(s.window):
            glfw.poll_events()
            s.impl.process_inputs()

            imgui.new_frame()

            if imgui.begin_main_menu_bar():
                if imgui.begin_menu("File", True):

                    clicked_quit, selected_quit = imgui.menu_item(
                        "Quit", "Cmd+Q", False, True
                    )

                    if clicked_quit:
                        sys.exit(0)

                    imgui.end_menu()
                imgui.end_main_menu_bar()


                imgui.set_next_window_position(0, 17)
                imgui.set_next_window_size_constraints(
                        (s.width, s.height),
                        (s.width, s.height),
                    )
                imgui.begin("Custom window", True, flags = imgui.WINDOW_NO_TITLE_BAR)
                imgui.text("Bar")
                imgui.text_ansi("B\033[31marA\033[mnsi ")
                imgui.text_ansi_colored("Eg\033[31mgAn\033[msi ", 0.2, 1.0, 0.0)
                imgui.extra.text_ansi_colored("Eggs", 0.2, 1.0, 0.0)
                imgui.end()

            # imgui.show_test_window()

            gl.glClearColor(0.0, 0.0, 0.0, 0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            imgui.render()
            s.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(s.window)



    def impl_glfw_init(self):
        width, height = 1280, 720
        window_name = "minimal ImGui/GLFW3 example"

        if not glfw.init():
            print("Could not initialize OpenGL context")
            sys.exit(1)

        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, gl.GL_TRUE)

        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(int(width), int(height), window_name, None, None)
        glfw.make_context_current(window)
        glfw.set_window_user_pointer(window, self)


        if not window:
            glfw.terminate()
            print("Could not initialize Window")
            sys.exit(1)
        glfw.set_framebuffer_size_callback(window, callback)

        return window
