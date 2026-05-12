import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

GRID = 80
STEPS = 200
INTERVAL = 60  # ms per frame

def step(grid):
    neighbors = sum(
        np.roll(np.roll(grid, i, 0), j, 1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((neighbors == 3) | (grid & (neighbors == 2))).astype(np.uint8)

def random_grid(n, density=0.3):
    return (np.random.rand(n, n) < density).astype(np.uint8)

grid = random_grid(GRID)

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.axis("off")
img = ax.imshow(grid, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
title = ax.set_title("Generation 0", color="white", fontsize=12)

def update(frame):
    global grid
    grid = step(grid)
    img.set_data(grid)
    title.set_text(f"Generation {frame + 1}")
    return img, title

ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=INTERVAL, blit=True)

out = "game_of_life.gif"
writer = animation.PillowWriter(fps=1000 // INTERVAL)
ani.save(out, writer=writer)
print(f"Saved {out}")
