import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import normalization

# More advanced visualizer based on the Open3D GetCoord Example.
class GUI:
    def __init__(self):
        # Create a SceneWidget that fills the entire window, and
        # a label in the lower left on top of the SceneWidget to display the
        # coordinate.
        app = gui.Application.instance
        self.window = app.create_window("Open3D - Multimedia Retrieval GUI", 1024, 768)
        # Since we want the label on top of the scene, we cannot use a layout,
        # so we need to manually layout the window's children.
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.Horiz()
        

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


def main():
    app = gui.Application.instance
    app.initialize()

    ex = GUI()

    app.run()


if __name__ == "__main__":
    main()


