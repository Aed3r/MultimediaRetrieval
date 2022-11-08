# TKinter interface with a button:

import time
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tracemalloc import start
import visualizer as vis
import open3d as o3d
import load_meshes as lm
from PIL import ImageTk, Image
import distance_functions as df
import normalization
import ShapeDescriptors
from tkinter.ttk import Progressbar

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("My App")
        self.geometry("1000x700")
        
        # Create a horizontal layout
        self.top_layout = ttk.Frame(self)
        self.top_layout.pack(side=LEFT, fill=BOTH, expand=True)

        # Create a vertical layout
        self.controls_layout = ttk.Frame(self.top_layout)
        self.controls_layout.pack(side=LEFT, fill=BOTH, expand=True)

        self.loadBtn = Button(self.controls_layout, text="Load a mesh", command=self.on_button_click)
        self.loadBtn.pack(pady=20)

        self.analyzeBtn = Button(self.controls_layout, text="Analyze", command=self.on_analyze_click)
        self.analyzeBtn["state"] = "disabled"
        self.analyzeBtn.pack(pady=20)

    def on_button_click(self):
        # Open a file dialog
        filename = filedialog.askopenfilename(
            title="Choose a file",
            filetypes=(("OFF files", "*.off"), ("PLY files", "*.ply"))
        )
        if not filename:
            return

        if hasattr(self, "vert_layout_loaded_img"):
            self.vert_layout_loaded_img.pack_forget()

        if hasattr(self, "vert_layout_desc_loaded"):
            self.vert_layout_desc_loaded.pack_forget()

        print("Loading mesh...")
        start = time.time()
        self.mesh = lm.load_mesh(filename, returnInfoOnly=False)
        print(f"Mesh loaded successfully. ({round(time.time() - start)}s)")

        print("Normalizing mesh...")
        start = time.time()
        # Normalize the input mesh
        self.mesh = normalization.normalize(self.mesh)
        end = time.time()
        print("Normalizing mesh done in ", end - start, " seconds")

        print("Extracting mesh features...")
        start = time.time()
        # get features from the input (querying) mesh
        self.mesh = ShapeDescriptors.extract_all_features(self.mesh, returnFullMesh=True)
        end = time.time()
        print("Extracting mesh features done in ", end - start, " seconds")

        thumb = vis.gen_thumbnails([self.mesh])[0]
        img = Image.open(thumb["thumbnailPath"])
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)

        # Create a vertical layout
        self.vert_layout_loaded_img = ttk.Frame(self.top_layout)
        self.vert_layout_loaded_img.pack(side=LEFT, fill=BOTH, expand=True)

        loadedThumb = Label(self.vert_layout_loaded_img, image=img)
        loadedThumb.image = img
        loadedThumb.pack()

        # Create a vertical layout
        self.vert_layout_desc_loaded = ttk.Frame(self.top_layout)
        self.vert_layout_desc_loaded.pack(side=LEFT, fill=BOTH, expand=True)

        # Name label
        label = ttk.Label(self.vert_layout_desc_loaded, text=self.mesh["name"])
        label.pack()

        # Number of vertices label
        numVertsLabel = ttk.Label(self.vert_layout_desc_loaded, text="Number of vertices: " + str(self.mesh["numVerts"]))
        numVertsLabel.pack()

        # Number of faces label
        numFacesLabel = ttk.Label(self.vert_layout_desc_loaded, text="Number of faces: " + str(self.mesh["numFaces"]))
        numFacesLabel.pack()

        # Volume label
        volumeLabel = ttk.Label(self.vert_layout_desc_loaded, text="Volume: " + str(self.mesh["volume"]))
        volumeLabel.pack()

        # Surface area label
        surfaceAreaLabel = ttk.Label(self.vert_layout_desc_loaded, text="Surface area: " + str(self.mesh["surface_area"]))
        surfaceAreaLabel.pack()

        # Compactness label
        compactnessLabel = ttk.Label(self.vert_layout_desc_loaded, text="Compactness: " + str(self.mesh["compactness"]))
        compactnessLabel.pack()

        # Diameter label
        diameterLabel = ttk.Label(self.vert_layout_desc_loaded, text="Diameter: " + str(self.mesh["diameter"]))
        diameterLabel.pack()

        # Eccentricity label
        eccentricityLabel = ttk.Label(self.vert_layout_desc_loaded, text="Eccentricity: " + str(self.mesh["eccentricity"]))
        eccentricityLabel.pack()

        # Rectangularity label
        rectangularityLabel = ttk.Label(self.vert_layout_desc_loaded, text="Rectangularity: " + str(self.mesh["rectangularity"]))
        rectangularityLabel.pack()

        self.analyzeBtn["state"] = "normal"

    def on_analyze_click(self):
        res = df.find_best_matches(self.mesh, 5)

        # Create a horizontal layout
        self.horiz_layout_res = ttk.Frame(self)
        self.horiz_layout_res.pack(side=LEFT, fill=BOTH, expand=True)

        # Disable analyze button
        self.analyzeBtn["state"] = "disabled"

        for resMesh in res:
            # Create a vertical layout
            vert_layout = ttk.Frame(self.horiz_layout_res)
            vert_layout.pack(side=LEFT, fill=BOTH, expand=True)
            
            if "thumbnailPath" in resMesh:
                img = Image.open(resMesh["thumbnailPath"])
            else:
                thumb = vis.gen_thumbnails([resMesh])[0]
                img = Image.open(thumb["thumbnailPath"])
            img = img.resize((250, 250), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)

            # Name label
            label = ttk.Label(vert_layout, text=resMesh["name"])
            label.pack()

            # Distance label
            distance = round((1 - resMesh["distance"][0]) * 100, 2)
            distLabel = ttk.Label(vert_layout, text=str(distance) + "%")
            distLabel.pack()

            panel = Label(vert_layout, image=img)
            panel.image = img
            panel.pack()

    def run(self):
        self.mainloop()
    
if __name__ == "__main__":
    app = App()
    app.run()