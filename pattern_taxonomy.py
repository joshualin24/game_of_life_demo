import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

PANEL_BG = "#1a1a1a"


def parse(spec: str):
    """Turn an ASCII pattern (#/O = live) into a list of (row, col) cells."""
    rows = spec.strip("\n").splitlines()
    return [(r, c) for r, row in enumerate(rows)
            for c, ch in enumerate(row) if ch in "#O"]


# Canonical patterns grouped by the taxonomy class they belong to.
PATTERNS = {
    # Still lifes — fixed points, never change.
    "block":   "##\n##",
    "beehive": ".##.\n#..#\n.##.",
    "loaf":    ".##.\n#..#\n.#.#\n..#.",
    "boat":    "##.\n#.#\n.#.",
    "tub":     ".#.\n#.#\n.#.",
    # Oscillators — return to their start after a fixed period.
    "blinker": "###",
    "toad":    ".###\n###.",
    "beacon":  "##..\n##..\n..##\n..##",
    "pulsar": ("..###...###..\n"
               ".............\n"
               "#....#.#....#\n"
               "#....#.#....#\n"
               "#....#.#....#\n"
               "..###...###..\n"
               ".............\n"
               "..###...###..\n"
               "#....#.#....#\n"
               "#....#.#....#\n"
               "#....#.#....#\n"
               ".............\n"
               "..###...###.."),
    "pentadecathlon": "..#....#..\n##.####.##\n..#....#..",
    # Spaceships — translate across the grid forever.
    "glider": ".#.\n..#\n###",
    "lwss":   ".#..#\n#....\n#...#\n####.",
    "mwss":   "..#...\n#...#.\n.....#\n#....#\n.#####",
    "hwss":   "...##..\n.#....#\n#......\n#.....#\n######.",
    # Methuselahs — tiny seeds that churn for hundreds of generations.
    "r_pentomino": ".##\n##.\n.#.",
}


class Board:
    """A toroidal Game of Life grid."""

    def __init__(self, size: int):
        self.size = size
        self.cells = np.zeros((size, size), dtype=np.uint8)

    def stamp(self, key: str, top: int, left: int):
        for r, c in parse(PATTERNS[key]):
            self.cells[top + r, left + c] = 1
        return self

    def step(self):
        neighbors = sum(
            np.roll(np.roll(self.cells, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        )
        self.cells = ((neighbors == 3) | (self.cells & (neighbors == 2))).astype(np.uint8)


class Panel:
    """One taxonomy class rendered in its own axes."""

    def __init__(self, ax, board: Board, title: str, subtitle: str, cell_color: str):
        self.board = board
        self.ax = ax
        ax.set_facecolor(PANEL_BG)
        ax.axis("off")
        cmap = ListedColormap([PANEL_BG, cell_color])
        self._img = ax.imshow(board.cells, cmap=cmap, vmin=0, vmax=1,
                              interpolation="nearest")
        ax.set_title(title, color="white", fontsize=12, pad=10, fontweight="bold")
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes, ha="center",
                va="top", color="#9aa", fontsize=8.5)

    def update(self):
        self.board.step()
        self._img.set_data(self.board.cells)
        return self._img


class PatternTaxonomyDemo:
    def __init__(self, steps: int = 180, fps: int = 15):
        self.steps = steps
        self.fps = fps

        self._fig, axes = plt.subplots(2, 2, figsize=(11, 11.6))
        self._fig.patch.set_facecolor("#0d0d0d")
        self._fig.suptitle(
            "Conway's Game of Life — A Pattern Taxonomy",
            color="white", fontsize=16, fontweight="bold", y=0.985,
        )

        self._panels = [
            Panel(axes[0, 0], self._build_still_lifes(),
                  "Still Lifes",
                  "block  ·  beehive  ·  loaf  ·  boat  ·  tub", "#4dd0e1"),
            Panel(axes[0, 1], self._build_oscillators(),
                  "Oscillators",
                  "blinker · toad · beacon · pulsar · pentadecathlon", "#9ccc65"),
            Panel(axes[1, 0], self._build_spaceships(),
                  "Spaceships",
                  "glider  ·  LWSS  ·  MWSS  ·  HWSS", "#ffb74d"),
            Panel(axes[1, 1], self._build_methuselahs(),
                  "Methuselahs",
                  "R-pentomino  (stabilizes after ~1100 generations)", "#ff7043"),
        ]

        self._gen_text = self._fig.text(
            0.5, 0.93, "", ha="center", color="#7fd6ff",
            fontsize=11, fontfamily="monospace",
        )
        self._fig.subplots_adjust(left=0.03, right=0.97, top=0.90,
                                  bottom=0.05, wspace=0.10, hspace=0.20)

    @staticmethod
    def _build_still_lifes() -> Board:
        return (Board(22)
                .stamp("block", 3, 3)
                .stamp("beehive", 3, 12)
                .stamp("loaf", 10, 3)
                .stamp("boat", 11, 13)
                .stamp("tub", 16, 8))

    @staticmethod
    def _build_oscillators() -> Board:
        return (Board(34)
                .stamp("blinker", 4, 4)
                .stamp("toad", 4, 24)
                .stamp("beacon", 16, 3)
                .stamp("pulsar", 10, 12)
                .stamp("pentadecathlon", 28, 10))

    @staticmethod
    def _build_spaceships() -> Board:
        return (Board(48)
                .stamp("glider", 4, 4)
                .stamp("lwss", 4, 24)
                .stamp("mwss", 24, 6)
                .stamp("hwss", 26, 26))

    @staticmethod
    def _build_methuselahs() -> Board:
        return Board(64).stamp("r_pentomino", 31, 31)

    def _update(self, frame):
        artists = [p.update() for p in self._panels]
        self._gen_text.set_text(f"Generation {frame + 1:3d}")
        return (*artists, self._gen_text)

    def run(self, output: str = "pattern_taxonomy.gif"):
        ani = animation.FuncAnimation(
            self._fig, self._update,
            frames=self.steps, interval=1000 // self.fps, blit=False,
        )
        ani.save(output, writer=animation.PillowWriter(fps=self.fps))
        print(f"Saved {output}")


if __name__ == "__main__":
    PatternTaxonomyDemo(steps=180, fps=15).run()
