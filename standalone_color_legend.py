"""Standalone Color Legend app"""
from tkinter import Canvas, Tk


def intensity_to_color(red: float, blue: float) -> str:
    """Convert intensity values into a color hex string."""
    delta: float = abs(red - blue)
    green: int = 175 + int(80 * delta) if (red or blue) else 175
    red = 175 + int(80 * red) if red else 175
    blue = 175 + int(80 * blue) if blue else 175
    return f"#{red:02x}{green:02x}{blue:02x}"


def inverse_intensity_to_color(red: float, blue: float) -> str:
    """Convert intensity values into a color hex string."""
    delta: float = abs(red - blue)
    green: int = 255 - int(175 * delta) if (red or blue) else 255
    red = 255 - int(175 * red) if red else 255
    blue = 255 - int(175 * blue) if blue else 255
    return f"#{red:02x}{green:02x}{blue:02x}"


def draw_color_grid(canvas: Canvas, grid_size: int = 50):
    """Draws a 2D grid of colors representing different move intensities."""
    width: int = 1000  # Canvas size
    cell_size: int = width // grid_size

    row: int
    for row in range(grid_size):
        col: int
        for col in range(grid_size):
            # Normalize x (White move dominance) and y (Black move dominance) to range [0,1]
            white_intensity: float = col / (grid_size - 1)  # More white moves → blue
            black_intensity: float = row / (grid_size - 1)  # More black moves → red

            # Get the color from our function
            color: str = intensity_to_color(black_intensity, white_intensity)

            if color == "#000000":
                color = "white"

            # Compute rectangle coordinates
            x0_: int = col * cell_size
            y0_: int = row * cell_size
            x1_: int = x0_ + cell_size
            y1_: int = y0_ + cell_size

            # Draw color cell
            canvas.create_rectangle(x0_, y0_, x1_, y1_, fill=color, outline="black")

    # Labels
    canvas.create_text(20, 0, text="Intensity of Possible White Moves →", anchor="nw", font=("Arial", 15), fill="white")
    # noinspection PyArgumentList
    canvas.create_text(0, 20, text="Intensity of Possible Black Moves →", anchor="sw", font=("Arial", 15),
                       angle=-90, fill="black")


if __name__ == "__main__":
    # Create Tkinter window
    root: Tk = Tk()
    root.title("Chess Heatmap 2D Color Grid")

    can: Canvas = Canvas(root, width=1000, height=1000, bg="white")
    can.pack()

    draw_color_grid(can, grid_size=50)  # Adjust grid_size for smoother gradient

    root.mainloop()
