"""Tooltips"""
from tkinter import Canvas, Event, Frame, LEFT, NSEW, SOLID, Tk, Toplevel, ttk
from tkinter.ttk import Label
from typing import Optional, Tuple

__all__ = [
    'CanvasTooltip',
]


class ChessHeatMapApp(Tk):
    """ChessHeatMapApp"""
    updating: bool


# pylint: disable=too-many-instance-attributes
class CanvasTooltip:
    """
    It creates a tooltip for a given canvas tag or id as the mouse is
    above it.

    This class has been derived from the original Tooltip class I updated
    and posted back to StackOverflow at the following link:

    https://stackoverflow.com/questions/3221956/what-is-the-simplest-way-to-make-tooltips-in-tkinter/41079350#41079350

    Alberto Vassena on 2016.12.10.
    """
    bg_color: str
    pad: Tuple[int, int, int, int]
    tw_: Optional[Toplevel]
    id_: Optional[str]
    text: str
    canvas: Canvas
    root: ChessHeatMapApp
    wraplength: int
    waittime: int
    font_size: int

    def __init__(
            self, root: ChessHeatMapApp, canvas: Canvas, tag_or_id: int,
            *,
            bg_color: str = '#FFFFEA',
            pad: Tuple[int, int, int, int] = (5, 3, 5, 3),
            text: str = 'canvas info',
            waittime: int = 150,
            wraplength: int = 500,
            font_size: int = 20
    ) -> None:
        self.font_size = font_size
        self.waittime = waittime  # in miliseconds, originally 500
        self.wraplength = wraplength  # in pixels, originally 180
        self.root = root
        self.canvas = canvas
        self.text = text
        self.canvas.tag_bind(tag_or_id, "<Enter>", self.onEnter)
        self.canvas.tag_bind(tag_or_id, "<Leave>", self.onLeave)
        self.canvas.tag_bind(tag_or_id, "<ButtonPress>", self.onLeave)
        self.bg_color = bg_color
        self.pad = pad
        self.id_ = None
        self.tw_ = None

    # pylint: disable=invalid-name
    def onEnter(self, _: Optional[Event] = None) -> None:
        """

        Parameters
        ----------
        _ : tkinter.Event
        """
        if not self.root.updating:
            self.schedule()

    # pylint: disable=invalid-name
    def onLeave(self, _: Optional[Event] = None) -> None:
        """

        Parameters
        ----------
        _ : tkinter.Event
        """
        self.unschedule()
        self.hide()

    def schedule(self) -> None:
        """schedule self"""
        self.unschedule()
        self.id_ = self.canvas.after(self.waittime, self.show)

    def unschedule(self) -> None:
        """unschedule self"""
        id_, self.id_ = self.id_, None
        if id_ is not None:
            self.canvas.after_cancel(id_)

    def show(self, _: Optional[Event] = None) -> None:
        """

        Parameters
        ----------
        _ : tkinter.Event

        """

        # pylint: disable=too-many-locals
        def tip_pos_calculator(
                canvas: Canvas,
                label: ttk.Label,
                *,
                tip_delta: Tuple[int, int] = (10, 5),
                pad: Tuple[int, int, int, int] = (5, 3, 5, 3)
        ) -> Tuple[int, int]:
            """

            Parameters
            ----------
            canvas : tkinter.Canvas
            label : tkinter.ttk.Label
            tip_delta : Tuple[int, int]
            pad : Tuple[int, int, int, int]

            Returns
            -------

            """
            c: Canvas = canvas
            s_width: int
            s_height: int
            s_width, s_height = c.winfo_screenwidth(), c.winfo_screenheight()
            width: int
            height: int
            width, height = (
                pad[0] + label.winfo_reqwidth() + pad[2],
                pad[1] + label.winfo_reqheight() + pad[3]
            )
            mouse_x: int
            mouse_y: int
            mouse_x, mouse_y = c.winfo_pointerxy()
            x1: int
            y1: int
            x1, y1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
            x2: int
            y2: int
            x2, y2 = x1 + width, y1 + height
            x_delta: int = max(x2 - s_width, 0)
            y_delta: int = max(y2 - s_height, 0)
            offscreen: bool = (x_delta, y_delta) != (0, 0)
            if offscreen:
                if x_delta:
                    x1 = mouse_x - tip_delta[0] - width
                if y_delta:
                    y1 = mouse_y - tip_delta[1] - height
            offscreen_again: bool = y1 < 0  # out on the top
            if offscreen_again:
                # No further checks will be done.
                # TIP:
                # A further mod might automagically augment the
                # wraplength when the tooltip is too high to be
                # kept inside the screen.
                y1 = 0
            return x1, y1

        if not self.root.updating:
            bg: str = self.bg_color
            pad: Tuple[int, int, int, int] = self.pad
            canvas: Canvas = self.canvas
            # creates a toplevel window
            self.tw_ = Toplevel(canvas.master)
            # Leaves only the label and removes the app window
            self.tw_.wm_overrideredirect(True)

            win: Frame = Frame(
                self.tw_,
                background=bg,
                borderwidth=0
            )
            label: Label = ttk.Label(
                win,
                text=self.text,
                justify=LEFT,
                background=bg,
                relief=SOLID,
                borderwidth=0,
                wraplength=self.wraplength,
                font=(self.root.font, self.font_size)
            )

            label.grid(
                padx=(pad[0], pad[2]),
                pady=(pad[1], pad[3]),
                sticky=NSEW
            )
            win.grid()
            x: int
            y: int
            x, y = tip_pos_calculator(canvas, label)
            self.tw_.wm_geometry(f"+{x}+{y}")

    def hide(self) -> None:
        """hide self"""
        if self.tw_ is not None:
            self.tw_.destroy()
            self.tw_ = None
