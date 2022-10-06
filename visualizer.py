import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import util

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
        self._geometry = o3d.geometry.TriangleMesh()
        self._geometry.vertices = o3d.utility.Vector3dVector(self._mesh.vertices)
        self._geometry.triangles = o3d.utility.Vector3iVector(self._mesh.faces)
        self._geometry.compute_vertex_normals()
        self.widget3d.scene.add_geometry("Mesh", self._geometry, self._plasticMat)

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
        self.wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        self._showWireframe = False

        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        #self.widget3d.look_at(center, center - [0, 0, 3], [0, -1, 0])

        self.widget3d.set_on_mouse(self._on_mouse_widget3d)
        self.widget3d.set_on_key(self._on_key_widget3d)

        self._shapeAligned = False
        self._aligned_mesh = util.align_shape(self._mesh)
        self._aligned_geometry = o3d.geometry.TriangleMesh()
        self._aligned_geometry.vertices = o3d.utility.Vector3dVector(self._aligned_mesh.vertices)
        self._aligned_geometry.triangles = o3d.utility.Vector3iVector(self._aligned_mesh.faces)
        self._aligned_geometry.compute_vertex_normals()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())

        prefShading = self.shadingLabel.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)
        self.shadingLabel.frame = gui.Rect(r.x, r.y, prefShading.width, prefShading.height)

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
        
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.W:
                if self._showWireframe:
                    # Remove wireframe
                    self.widget3d.scene.remove_geometry("Wireframe")
                    self._showWireframe = False
                else:
                    # Add wireframe
                    self.widget3d.scene.add_geometry("Wireframe", self.wireframe, self.wireframeMat)
                    self._showWireframe = True
            elif event.key == gui.S:
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
                    if self._shapeAligned:
                        self.widget3d.scene.add_geometry("Mesh", self._aligned_geometry, self._plasticMat)
                    else:
                        self.widget3d.scene.add_geometry("Mesh", self._geometry, self._plasticMat)

                if self._showWireframe:
                    self.widget3d.scene.remove_geometry("Wireframe")
                self.widget3d.scene.update_material(self._plasticMat)
                if self._showWireframe:
                    self.widget3d.scene.add_geometry("Wireframe", self.wireframe, self.wireframeMat)

                # Update label
                update_shadingLabel()
            elif event.key == gui.B:
                # Switch skybox on or off
                self.widget3d.scene.show_skybox(not self._showSkybox)
                self._showSkybox = not self._showSkybox
            elif event.key == gui.A:
                # Switch axis on or off
                self.widget3d.scene.show_axes(not self._showAxes)
                self._showAxes = not self._showAxes
            elif event.key == gui.Y:
                # Switch shadow options
                self._shadowIndex = (self._shadowIndex + 1) % len(self._shadowOptions)
                self.widget3d.scene.set_lighting(self._shadowOptions[self._shadowIndex], self._sunDir)

                # Update label
                update_shadingLabel()
            elif event.key == gui.F:
                # Align the shape
                if self._shapeAligned:
                    self.widget3d.scene.remove_geometry("AlignedMesh")
                    self.widget3d.scene.add_geometry("Mesh", self._geometry, self._plasticMat)
                    self._shapeAligned = False
                else:
                    self.widget3d.scene.remove_geometry("Mesh")
                    self.widget3d.scene.add_geometry("AlignedMesh", self._aligned_geometry, self._plasticMat)
                    self._shapeAligned = True
                            
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
    print("-- Visual options --")
    print("  W            : Toggle wireframe.")
    print("  S            : Switch shading option.")
    print("  B            : Toggle skybox.")
    print("  A            : Toggle axis.")
    print("  Y            : Switch lighting options.")
    print("  F            : Align the shape.")
    print("")
    print("-- General control --")
    print("  Q, Esc       : Exit window.")
    print("  H            : Print help message.")
    print("  P, PrtScn    : Take a screen capture.")
    print("  D            : Take a depth capture.")
    print("  O            : Take a capture of current rendering settings.")

def main():
    import load_meshes

    app = gui.Application.instance
    app.initialize()

    print_help()

    #armadillo_mesh = o3d.data.ArmadilloMesh()
    #mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)

    random_mesh = load_meshes.get_meshes(True, True, 1, False)[0]
    ex = MMRVISUALIZER(random_mesh)

    app.run()


if __name__ == "__main__":
    main()


