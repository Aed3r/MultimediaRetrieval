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
        self.loadBtn = Button(self, text="Load a mesh", command=self.on_button_click)
        self.loadBtn.pack()

    def on_button_click(self):
        # Open a file dialog
        filename = filedialog.askopenfilename(
            title="Choose a file",
            filetypes=(("OFF files", "*.off"), ("PLY files", "*.ply"))
        )
        if not filename:
            return

        # Remove the layout if they exist
        if hasattr(self, "horiz_layout_loaded"):
            self.horiz_layout_loaded.pack_forget()
        
        if hasattr(self, "horiz_layout_res"):
            self.horiz_layout_res.pack_forget()

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

        # Create a horizontal layout
        self.horiz_layout_loaded = ttk.Frame(self)
        self.horiz_layout_loaded.pack(fill=BOTH, expand=True)

        # Create a vertical layout
        vert_layout_loaded_img = ttk.Frame(self.horiz_layout_loaded)
        vert_layout_loaded_img.pack(side=LEFT, fill=BOTH, expand=True)

        panel = Label(vert_layout_loaded_img, image=img)
        panel.image = img
        panel.pack()

        # Create a vertical layout
        vert_layout_desc_loaded = ttk.Frame(self.horiz_layout_loaded)
        vert_layout_desc_loaded.pack(side=LEFT, fill=BOTH, expand=True)

        # Name label
        label = ttk.Label(vert_layout_desc_loaded, text=self.mesh["name"])
        label.pack()

        # Number of vertices label
        numVertsLabel = ttk.Label(vert_layout_desc_loaded, text="Number of vertices: " + str(self.mesh["numVerts"]))
        numVertsLabel.pack()

        # Number of faces label
        numFacesLabel = ttk.Label(vert_layout_desc_loaded, text="Number of faces: " + str(self.mesh["numFaces"]))
        numFacesLabel.pack()

        # Volume label
        volumeLabel = ttk.Label(vert_layout_desc_loaded, text="Volume: " + str(self.mesh["volume"]))
        volumeLabel.pack()

        # Surface area label
        surfaceAreaLabel = ttk.Label(vert_layout_desc_loaded, text="Surface area: " + str(self.mesh["surface_area"]))
        surfaceAreaLabel.pack()

        # Compactness label
        compactnessLabel = ttk.Label(vert_layout_desc_loaded, text="Compactness: " + str(self.mesh["compactness"]))
        compactnessLabel.pack()

        # Diameter label
        diameterLabel = ttk.Label(vert_layout_desc_loaded, text="Diameter: " + str(self.mesh["diameter"]))
        diameterLabel.pack()

        # Eccentricity label
        eccentricityLabel = ttk.Label(vert_layout_desc_loaded, text="Eccentricity: " + str(self.mesh["eccentricity"]))
        eccentricityLabel.pack()

        # Rectangularity label
        rectangularityLabel = ttk.Label(vert_layout_desc_loaded, text="Rectangularity: " + str(self.mesh["rectangularity"]))
        rectangularityLabel.pack()

    def on_analyze_click(self):
        res = df.find_best_matches(self.mesh, 5)

        # Hide the buttons
        self.analyzeBtn.pack_forget()
        self.loadBtn.pack_forget()

        # Create a horizontal layout
        self.horiz_layout_res = ttk.Frame(self)
        self.horiz_layout_res.pack(fill=BOTH, expand=True)

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
            distance = round((1 - resMesh["distance"]) * 100, 2)
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