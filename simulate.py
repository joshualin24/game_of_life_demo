import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Grid:
    def __init__(self, size: int, density: float = 0.3):
        self.size = size
        self.cells = (np.random.rand(size, size) < density).astype(np.uint8)

    def step(self):
        neighbors = sum(
            np.roll(np.roll(self.cells, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        )
        self.cells = ((neighbors == 3) | (self.cells & (neighbors == 2))).astype(np.uint8)


class GameOfLife:
    def __init__(self, grid_size: int = 80, steps: int = 200, fps: int = 16):
        self.grid = Grid(grid_size)
        self.steps = steps
        self.fps = fps
        self._fig, self._ax, self._img, self._title = self._setup_figure()

    def _setup_figure(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.axis("off")
        img = ax.imshow(self.grid.cells, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        title = ax.set_title("Generation 0", color="white", fontsize=12)
        return fig, ax, img, title

    def _update(self, frame):
        self.grid.step()
        self._img.set_data(self.grid.cells)
        self._title.set_text(f"Generation {frame + 1}")
        return self._img, self._title

    def run(self, output: str = "game_of_life.gif"):
        ani = animation.FuncAnimation(
            self._fig, self._update,
            frames=self.steps, interval=1000 // self.fps, blit=True
        )
        ani.save(output, writer=animation.PillowWriter(fps=self.fps))
        print(f"Saved {output}")


if __name__ == "__main__":
    GameOfLife(grid_size=80, steps=200, fps=16).run()
