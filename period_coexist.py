import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LogisticMap:
    def __init__(self, r: float, x0: float = 0.5):
        self.r = r
        self.x = x0

    def f(self, x: float) -> float:
        return self.r * x * (1 - x)

    def step(self) -> float:
        self.x = self.f(self.x)
        return self.x

    def warmup(self, n: int = 2000):
        for _ in range(n):
            self.step()

    def detect_period(self, max_p: int = 30, tol: float = 1e-8) -> int:
        x0 = self.x
        for p in range(1, max_p + 1):
            xp = x0
            for _ in range(p):
                xp = self.f(xp)
            if abs(xp - x0) < tol:
                return p
        return -1

    @classmethod
    def find_r_for_period(cls, target: int, tol: float = 1e-8) -> float:
        search_ranges = {
            3: [(3.820, 3.860)],
            5: [(3.730, 3.760), (3.900, 3.930)],
        }
        for lo, hi in search_ranges.get(target, [(3.5, 4.0)]):
            for r in np.linspace(lo, hi, 6000):
                m = cls(r)
                m.warmup(2000)
                if m.detect_period(tol=tol) == target:
                    return float(r)
        raise RuntimeError(f"Period-{target} window not found")


class CobwebAnimator:
    def __init__(self, ax, lmap: LogisticMap, color: str, title: str):
        self.ax = ax
        self.lmap = lmap
        self._cx = [lmap.x]
        self._cy = [0.0]
        self._setup(color, title)

    def _setup(self, color: str, title: str):
        xs = np.linspace(0, 1, 500)
        self.ax.plot(xs, [self.lmap.f(x) for x in xs], color=color, lw=2.2)
        self.ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.35)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_facecolor("#0d0d0d")
        self.ax.set_title(title, color="white", fontsize=12, pad=8)
        self.ax.set_xlabel("xₙ", color="#aaa", fontsize=9)
        self.ax.set_ylabel("xₙ₊₁", color="#aaa", fontsize=9)
        self.ax.tick_params(colors="#888", labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#333")
        self._line, = self.ax.plot([], [], color="cyan", lw=0.9, alpha=0.85)

    def advance(self):
        x = self._cx[-1]
        y = self.lmap.f(x)
        self._cx += [x, y]
        self._cy += [y, y]
        self.lmap.x = y

    def draw(self):
        self._line.set_data(self._cx, self._cy)
        return self._line


class PeriodCoexistSim:
    _COLORS = {3: "#ff6b6b", 5: "#51cf66"}

    def __init__(self, steps: int = 120):
        self.steps = steps

        print("Searching for period-3 r...")
        r3 = LogisticMap.find_r_for_period(3)
        print(f"  Found r = {r3:.6f}")
        print("Searching for period-5 r...")
        r5 = LogisticMap.find_r_for_period(5)
        print(f"  Found r = {r5:.6f}")

        m3 = LogisticMap(r3)
        m3.warmup(2000)
        m5 = LogisticMap(r5)
        m5.warmup(2000)

        self._fig, (ax3, ax5) = plt.subplots(1, 2, figsize=(11, 5))
        self._fig.patch.set_facecolor("#0d0d0d")
        self._fig.suptitle(
            "Coexistence of Period-3 & Period-5\n"
            r"Logistic Map:  $x_{n+1} = r \cdot x_n (1 - x_n)$",
            color="white", fontsize=13, y=1.01,
        )

        self._cw3 = CobwebAnimator(ax3, m3, self._COLORS[3],
                                    f"Period-3   r = {r3:.5f}")
        self._cw5 = CobwebAnimator(ax5, m5, self._COLORS[5],
                                    f"Period-5   r = {r5:.5f}")
        self._fig.tight_layout()

    def _update(self, frame):
        self._cw3.advance()
        self._cw5.advance()
        return self._cw3.draw(), self._cw5.draw()

    def run(self, output: str = "period_coexist.gif"):
        ani = animation.FuncAnimation(
            self._fig, self._update,
            frames=self.steps, interval=80, blit=True,
        )
        ani.save(output, writer=animation.PillowWriter(fps=12))
        print(f"Saved {output}")


if __name__ == "__main__":
    PeriodCoexistSim(steps=120).run()
