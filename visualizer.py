import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import normalization

# More advanced visualizer based on the Open3D GetCoord Example.
class MMRVISUALIZER:
    def __init__(self, mesh):
        # Create a SceneWidget that fills the entire window, and
        # a label in the lower left on top of the SceneWidget to display the
        # coordinate.
        app = gui.Application.instance
        self.window = app.create_window("Open3D - Multimedia Retrieval Visualizer", 1024, 768)
        # Since we want the label on top of the scene, we cannot use a layout,
        # so we need to manually layout the window's children.
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        self.shadingLabel = gui.Label("")
        self.shadingLabel.visible = False
        self.window.add_child(self.shadingLabel)

        self.normLabel = gui.Label("")
        self.normLabel.visible = False
        self.window.add_child(self.normLabel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

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
        self._shadingOptions = ["defaultUnlit", "defaultLit", "normals", "depth", "none"]

        # Add mesh with material
        self._mesh = mesh
        self._ogVertices = o3d.utility.Vector3dVector(self._mesh["vertices"])
        self._ogFaces = o3d.utility.Vector3iVector(self._mesh["faces"])
        self._geometry = o3d.geometry.TriangleMesh()
        self._geometry.vertices = self._ogVertices
        self._geometry.triangles = self._ogFaces
        self._geometry.compute_triangle_normals()
        self.widget3d.scene.add_geometry("Mesh", self._geometry, self._plasticMat)

        # Normalized mesh
        self._norm_mesh = self._mesh.copy()
        self._normalizationSteps = ["None", "Remeshed", "Translated", "Posed", "Flipped", "Scaled"]
        self._normalizationStep = 0

        # Set lighting
        self.widget3d.scene.set_background([1.0000, 1.0000, 0.9294, 1.0000])
        self.widget3d.scene.show_skybox(True)
        self._showSkybox = True
        self.widget3d.scene.show_axes(False)
        self._showAxes = False
        self.widget3d.scene.scene.enable_indirect_light(True)
        self.widget3d.scene.scene.set_indirect_light_intensity(45000)
        self._sunDir = [0.577, -0.577, -0.577]
        self.widget3d.scene.scene.set_sun_light(self._sunDir, [1, 1, 1], 45000)
        self.widget3d.scene.scene.enable_sun_light(True)
        self._shadowOptions = [self.widget3d.scene.DARK_SHADOWS, self.widget3d.scene.HARD_SHADOWS, self.widget3d.scene.MED_SHADOWS, self.widget3d.scene.NO_SHADOWS, self.widget3d.scene.SOFT_SHADOWS]
        self._shadowIndex = 2
        self.widget3d.scene.set_lighting(self._shadowOptions[self._shadowIndex], self._sunDir)

        # Create Wireframe
        self.wireframeMat = rendering.MaterialRecord()
        self.wireframeMat.shader = "defaultUnlit"
        self.wireframeMat.base_color = [0.0, 0.0, 0.0, 1.0]
        self._showWireframe = False
        self.update_wireframe()
        
        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        #self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])

        self.widget3d.set_on_mouse(self._on_mouse_widget3d)
        self.widget3d.set_on_key(self._on_key_widget3d)

    def update_wireframe(self):
        self.wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self._geometry)
        
        if self._showWireframe:
            # Remove current wireframe
            self.widget3d.scene.remove_geometry("Wireframe")
            self.widget3d.scene.add_geometry("Wireframe", self.wireframe, self.wireframeMat)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())

        prefShading = self.shadingLabel.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())

        prefNorm = self.normLabel.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())

        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)
        self.shadingLabel.frame = gui.Rect(r.x, r.y, prefShading.width, prefShading.height)
        self.normLabel.frame = gui.Rect(r.get_right() - prefNorm.width, r.y, prefNorm.width, prefNorm.height)




    def _on_mouse_widget3d(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget.
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                else:
                    world = self.widget3d.scene.camera.unproject(
                        event.x, event.y, depth, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])

                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")
                    # We are sizing the info label to be exactly the right size,
                    # so since the text likely changed width, we need to
                    # re-layout to set the new frame.
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key_widget3d(self, event):
        def update_shadingLabel():
            text = "Shading: " + self._plasticMat.shader + "\nShadows: " + self._shadowOptions[self._shadowIndex].name
            self.shadingLabel.text = text
            self.shadingLabel.visible = (text != "")
            self.window.set_needs_layout()

        def update_normLabel():
            text = "Normalization step: " + self._normalizationSteps[self._normalizationStep]
            self.normLabel.text = text
            self.normLabel.visible = (text != "")
            self.window.set_needs_layout()
        
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.F1:
                if self._showWireframe:
                    # Remove wireframe
                    self.widget3d.scene.remove_geometry("Wireframe")
                    self._showWireframe = False
                else:
                    # Add wireframe
                    self.widget3d.scene.add_geometry("Wireframe", self.wireframe, self.wireframeMat)
                    self._showWireframe = True
            elif event.key == gui.F2:
                # Switch shading option
                self._plasticMat.shader = self._shadingOptions[
                    (self._shadingOptions.index(self._plasticMat.shader) + 1) %
                    len(self._shadingOptions)]

                if self._plasticMat.shader == "none":
                    # Remove mesh
                    self.widget3d.scene.remove_geometry("Mesh")
                    update_shadingLabel()
                    return gui.Widget.EventCallbackResult.HANDLED
                elif self._plasticMat.shader == "defaultUnlit":
                    # Add mesh with unlit material
                    self.widget3d.scene.add_geometry("Mesh", self._geometry, self._plasticMat)

                self.widget3d.scene.update_material(self._plasticMat)
                self.update_wireframe()
                
                # Update label
                update_shadingLabel()
            elif event.key == gui.F3:
                # Switch skybox on or off
                self.widget3d.scene.show_skybox(not self._showSkybox)
                self._showSkybox = not self._showSkybox
            elif event.key == gui.F4:
                # Switch axis on or off
                self.widget3d.scene.show_axes(not self._showAxes)
                self._showAxes = not self._showAxes
            elif event.key == gui.F5:
                # Switch shadow options
                self._shadowIndex = (self._shadowIndex + 1) % len(self._shadowOptions)
                self.widget3d.scene.set_lighting(self._shadowOptions[self._shadowIndex], self._sunDir)

                # Update label
                update_shadingLabel()
            elif event.key == gui.F6:
                # Go through normalization steps
                if self._normalizationStep == 0:
                    # Remeshing
                    self._norm_mesh = normalization.resampling(self._norm_mesh)  
                    self._geometry.vertices = o3d.utility.Vector3dVector(self._norm_mesh["vertices"])
                    self._geometry.triangles = o3d.utility.Vector3iVector(self._norm_mesh["faces"])   
                    self._normalizationStep = 1
                elif self._normalizationStep == 1:
                    # Translation
                    self._norm_mesh = normalization.translate_mesh_to_origin(self._norm_mesh)
                    self._geometry.vertices = o3d.utility.Vector3dVector(self._norm_mesh["vertices"])
                    self._geometry.triangles = o3d.utility.Vector3iVector(self._norm_mesh["faces"])   
                    self._normalizationStep = 2
                elif self._normalizationStep == 2:
                    # Pose
                    self._norm_mesh = normalization.align_shape(self._norm_mesh)
                    self._geometry.vertices = o3d.utility.Vector3dVector(self._norm_mesh["vertices"])
                    self._geometry.triangles = o3d.utility.Vector3iVector(self._norm_mesh["faces"])   
                    self._normalizationStep = 3
                elif self._normalizationStep == 3:
                    # Flipping
                    self._norm_mesh = normalization.flipping_test(self._norm_mesh)
                    self._geometry.vertices = o3d.utility.Vector3dVector(self._norm_mesh["vertices"])
                    self._geometry.triangles = o3d.utility.Vector3iVector(self._norm_mesh["faces"])   
                    self._normalizationStep = 4
                elif self._normalizationStep == 4:
                    # Scale
                    self._norm_mesh = normalization.scale_mesh_to_unit(self._norm_mesh)
                    self._geometry.vertices = o3d.utility.Vector3dVector(self._norm_mesh["vertices"])
                    self._geometry.triangles = o3d.utility.Vector3iVector(self._norm_mesh["faces"])   
                    self._normalizationStep = 5
                elif self._normalizationStep == 5:
                    # Back to original mesh
                    self._geometry.vertices = self._ogVertices
                    self._geometry.triangles = self._ogFaces
                    self._norm_mesh = self._mesh.copy()
                    self._normalizationStep = 0
                
                update_normLabel()
                self.update_wireframe()
                self._geometry.compute_triangle_normals()
                self.widget3d.scene.remove_geometry("Mesh")
                if self._plasticMat.shader != "none":
                    self.widget3d.scene.add_geometry("Mesh", self._geometry, self._plasticMat)
                    

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED
        

def print_help():
    print("-- Mouse view control --")
    print("  Left button + drag         : Rotate.")
    print("  Ctrl + left button + drag  : Translate.")
    print("  Wheel button + drag        : Translate.")
    print("  Shift + left button + drag : Roll.")
    print("  Wheel                      : Zoom in/out.")
    print("")
    print("-- Keyboard view control --")
    print("  [/]          : Increase/decrease field of view.")
    print("  R            : Reset view point.")
    print("  Ctrl/Cmd + C : Copy current view status into the clipboard.")
    print("  Ctrl/Cmd + V : Paste view status from clipboard.")
    print("")
    print("-- Visual and mesh options --")
    print("  F1            : Toggle wireframe.")
    print("  F2            : Switch shading option.")
    print("  F3            : Toggle skybox.")
    print("  F4            : Toggle axis.")
    print("  F5            : Switch lighting options.")
    print("  F6            : Normalisation steps.")
    print("")
    print("-- General control --")
    print("  Q, Esc       : Exit window.")
    print("  H            : Print help message.")
    print("  P, PrtScn    : Take a screen capture.")
    print("  D            : Take a depth capture.")
    print("  O            : Take a capture of current rendering settings.")

def main():
    import load_meshes
    import sys
    import os

    app = gui.Application.instance
    app.initialize()

    print_help()

    # Check if there is a command line argument
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]

        # Check if the file exists
        if not os.path.isfile(mesh_path):
            raise Exception("File does not exist: {}".format(mesh_path))

        # Load the mesh
        if mesh_path.endswith('.off'):
            mesh = load_meshes.load_OFF(mesh_path, False)
        elif mesh_path.endswith('.ply'):
            mesh = load_meshes.load_PLY(mesh_path, False)
        else:
            raise Exception("File format not supported")
    else:
        mesh = load_meshes.get_meshes(True, True, 1, False)[0]

    ex = MMRVISUALIZER(mesh)

    app.run()


if __name__ == "__main__":
    main()


