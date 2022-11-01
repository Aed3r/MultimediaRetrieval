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
import distance_functions as df

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

        print("Loading mesh...")
        self.mesh = lm.load_mesh(filename, returnInfoOnly=False)
        print("Mesh loaded successfully.")
        print("Generating thumbnail...")
        thumb = vis.gen_thumbnails([self.mesh])[0]
        print("Thumbnail generated successfully.")
        img = Image.open(thumb["thumbnailPath"])
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self, image=img)
        panel.image = img
        panel.pack()

        # Load a "analyze" button
        self.button = Button(self, text="Analyze", command=self.on_analyze_click)
        self.button.pack()

    def on_analyze_click(self):
        res = df.find_best_matches(self.mesh, 5)

        # Create a horizontal layout
        self.horiz_layout = ttk.Frame(self)
        self.horiz_layout.pack(fill=BOTH, expand=True)

        for resMesh in res:
            if "thumbnailPath" in resMesh:
                img = Image.open(resMesh["thumbnailPath"])
            else:
                thumb = vis.gen_thumbnails([resMesh])[0]
                img = Image.open(thumb["thumbnailPath"])
            img = img.resize((250, 250), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            panel = Label(self.horiz_layout, image=img)
            panel.image = img
            panel.pack(side=LEFT)

    def run(self):
        self.mainloop()
    
if __name__ == "__main__":
    app = App()
    app.run()