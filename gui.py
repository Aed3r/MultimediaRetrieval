import glob
import math
import threading
from time import sleep
import time
from types import coroutine
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import asyncio
import database as db
import load_meshes as lm


class AppWindow:
    def __init__(self, width, height):
        resource_path = gui.Application.instance.resource_path

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # Mesh info
        #self._dbmngr = db.DatabaseManager()
        self._items = []
        #self._itemCount = self._dbmngr.get_mesh_count()
        self._itemCount = 30

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Grid
        self._itemsPerRow = 6
        self._grid = []

        # Controls Panel
        self._controls_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Visualizer for the loaded mesh
        # self._loadedMeshVis = gui.SceneWidget()
        # self._loadedMeshVis.scene = rendering.Open3DScene(w.renderer)
        # self._loadedMeshVis.background_color = gui.Color(1, 0, 0)
        # self._sunDir = [0.577, -0.577, -0.577]
        # self._loadedMeshVis.scene.set_lighting(self._loadedMeshVis.scene.MED_SHADOWS, self._sunDir)
        # self._loadedMeshVis.scene.show_skybox(False)
        # self._loadedMeshVis.scene.scene.enable_indirect_light(True)
        # self._loadedMeshVis.scene.scene.set_indirect_light_intensity(45000)
        # self._loadedMeshVis.scene.scene.set_sun_light(self._sunDir, [1, 1, 1], 45000)
        # self._loadedMeshVis.scene.scene.enable_sun_light(True)
        self._loadedMeshVis = gui.ImageWidget()

        # Load mesh button
        self._loadMeshButton = gui.Button("Load Mesh")
        self._loadMeshButton.set_on_clicked(self._on_load_mesh)
        self._controls_panel.add_child(self._loadMeshButton)

        self._controls_panel.add_fixed(separation_height)

        # Mesh Info
        self._meshInfo = gui.Label("")
        self._controls_panel.add_child(self._meshInfo)

        w.set_on_layout(self._on_layout)

        w.add_child(self._controls_panel)
        w.add_child(self._loadedMeshVis)

        # Set material
        self._plasticMat = rendering.MaterialRecord()
        self._plasticMat.base_color = [0.9765, 0.702, 0.0000, 1.0000]
        self._plasticMat.base_metallic = 0.0
        self._plasticMat.base_roughness = 0.5
        self._plasticMat.base_reflectance = 0.5
        self._plasticMat.base_clearcoat = 0.5
        self._plasticMat.base_clearcoat_roughness = 0.2
        self._plasticMat.base_anisotropy = 0.0
        self._plasticMat.shader = "defaultLit"

        # Lazy load the items into the grid
        def thread_main():
            self._itemCursor = self._dbmngr.get_all()

            while self._itemCursor.alive:
                # We can only modify GUI objects on the main thread, so we
                # need to post the function to call to the main thread.
                gui.Application.instance.post_to_main_thread(w, self._create_item)
                time.sleep(1)

        #threading.Thread(target=thread_main).start()

    def _create_item(self):
        meshItem = self._itemCursor.next()
        self._items.append(meshItem["_id"])
        item = gui.Button(meshItem['name'])
        item.set_on_clicked(lambda _ : self._on_item_clicked(meshItem["_id"]))
        #mesh = lm.load_mesh(item['path'], True)
        self._gridPanel.add_child(item)
        self.window.post_redraw()

    def _on_item_clicked(self, id):
        print(f"Item {str(id)} clicked")

    def _on_load_mesh(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_load_mesh_cancel)
        dlg.set_on_done(self._on_load_mesh_done)
        self.window.show_dialog(dlg)

    def _on_load_mesh_cancel(self):
        self.window.close_dialog()

    def _on_load_mesh_done(self, filename):
        self.window.close_dialog()
        self._loadedMesh = lm.load_mesh(filename, True)

        thumbpath = self._loadedMesh["path"] + ".png"
        try:
            thumb = lm.load_thumbnail(thumbpath)
        except:
            thumb = None

        # self._loadedGeometry = o3d.geometry.TriangleMesh()
        # self._loadedGeometry.vertices = o3d.utility.Vector3dVector(self._loadedMesh["vertices"])
        # self._loadedGeometry.triangles = o3d.utility.Vector3iVector(self._loadedMesh["faces"])
        # self._loadedGeometry.compute_vertex_normals()
        # self._loadedGeometry.compute_triangle_normals()
        #self._loadedMeshVis.scene.add_geometry("Mesh", self._loadedGeometry, self._plasticMat)
        self._loadedMeshVis.update_image(thumb)

    def _set_grid_item_settings(self, item):
        item.scene.set_background(gui.Color(0, 0, 0))
        item.scene.show_skybox(False)
        item.scene.scene.enable_indirect_light(True)
        item.scene.scene.set_indirect_light_intensity(45000)
        self._sunDir = [0.577, -0.577, -0.577]
        item.scene.scene.set_sun_light(self._sunDir, [1, 1, 1], 45000)
        item.scene.scene.enable_sun_light(True)
        # [item.scene.DARK_SHADOWS, item.scene.HARD_SHADOWS, item.scene.MED_SHADOWS, item.scene.NO_SHADOWS, item.scene.SOFT_SHADOWS]
        item.scene.set_lighting(item.scene.MED_SHADOWS, self._sunDir)

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        
        #min(r.height, self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        #self._settings_panel.frame = gui.Rect(r.get_right() - settingsWidth, r.y, settingsWidth, settingsHeight)

        r = self.window.content_rect
        em = self.window.theme.font_size
        sep = 0.5 * em

        # Controls panel
        controlsWidth = 17 * layout_context.theme.font_size
        controlsHeight = r.height / 3
        self._controls_panel.frame = gui.Rect(r.get_right() - controlsWidth, r.y, controlsWidth, controlsHeight)

        # Load mesh visualizer
        loadedMeshVisWidth = controlsWidth
        loadedMeshVisHeight = controlsWidth
        self._loadedMeshVis.frame = gui.Rect(r.get_right() - controlsWidth, controlsHeight, loadedMeshVisWidth, loadedMeshVisHeight)


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()