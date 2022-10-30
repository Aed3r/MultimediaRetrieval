# TKinter interface with a button:

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import visualizer as vis
import open3d as o3d
import load_meshes as lm
from PIL import ImageTk, Image

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("My App")
        self.geometry("1000x1000")
        self.button = Button(self, text="Load a mesh", command=self.on_button_click)
        self.button.pack()

    def on_button_click(self):
        # Open a file dialog
        filename = filedialog.askopenfilename(
            title="Choose a file",
            filetypes=(("OFF files", "*.off"), ("PLY files", "*.ply"))
        )
        if not filename:
            return

        mesh = lm.load_mesh(filename, returnInfoOnly=False)
        img = Image.open(vis.gen_thumbnail(mesh))
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self, image=img)
        panel.image = img
        panel.pack()

    def run(self):
        self.mainloop()
    
if __name__ == "__main__":
    app = App()
    app.run()