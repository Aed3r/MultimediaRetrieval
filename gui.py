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
        w = self.window  # to make the code more concise
        em = w.theme.font_size # font size in pixels

        # Top level separation
        self.mainPanel = gui.Horiz(spacing=0, margins=gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.mainPanel.add_stretch()
        self.mainPanel.background_color = gui.Color(0, 0, 1, 1.0)

        # Left panel: grid
        self.gridPanel = gui.VGrid(cols=5, spacing=0.2 * em, margins=gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.gridPanel.background_color = gui.Color(0, 1, 0, 1.0)
        self.mainPanel.add_child(self.gridPanel)

        # Right panel: controls
        self.controlsPanel = gui.Vert(spacing=0.2 * em, margins=gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.controlsPanel.background_color = gui.Color(1, 0, 0, 1.0)
        self.controlsPanel.add_fixed(30 * em)
        self.controlsPanel.add_child(gui.Label("Controls"))
        self.mainPanel.add_child(self.controlsPanel)

        self.window.set_on_layout(self._on_layout)



    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r


def main():
    app = gui.Application.instance
    app.initialize()

    ex = GUI()

    app.run()


if __name__ == "__main__":
    main()


