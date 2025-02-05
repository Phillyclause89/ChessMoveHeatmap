import tkinter as tk


def intensity_to_color(red: float, blue: float) -> str:
    """Convert intensity values into a color hex string."""
    delta = abs(red - blue)
    green = 175 + int(80 * delta) if (red or blue) else 175
    red = 175 + int(80 * red) if red else 175
    blue = 175 + int(80 * blue) if blue else 175
    return f"#{red:02x}{green:02x}{blue:02x}"


def inverse_intensity_to_color(red: float, blue: float) -> str:
    """Convert intensity values into a color hex string."""
    delta = abs(red - blue)
    green = 255 - int(175 * delta) if (red or blue) else 255
    red = 255 - int(175 * red) if red else 255
    blue = 255 - int(175 * blue) if blue else 255
    return f"#{red:02x}{green:02x}{blue:02x}"


def draw_color_grid(canvas: tk.Canvas, grid_size: int = 50):
    """Draws a 2D grid of colors representing different move intensities."""
    width = 800  # Canvas size
    cell_size = width // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            # Normalize x (White move dominance) and y (Black move dominance) to range [0,1]
            white_intensity = col / (grid_size - 1)  # More white moves → blue
            black_intensity = row / (grid_size - 1)  # More black moves → red

            # Get the color from our function
            color = intensity_to_color(black_intensity, white_intensity)

            if color == "#000000":
                color = "white"

            # Compute rectangle coordinates
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            # Draw color cell
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")

    # Labels
    canvas.create_text(20, 0, text="Intensity of Possible White Moves →", anchor="nw", font=("Arial", 10), fill="white")
    canvas.create_text(0, 20, text="Intensity of Possible Black Moves →", anchor="sw", font=("Arial", 10),
                       angle=-90, fill="black")


if __name__ == "__main__":
    # Create Tkinter window
    root = tk.Tk()
    root.title("Chess Heatmap 2D Color Grid")

    canvas = tk.Canvas(root, width=800, height=800, bg="white")
    canvas.pack()

    draw_color_grid(canvas, grid_size=50)  # Adjust grid_size for smoother gradient

    root.mainloop()
