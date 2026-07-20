from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


# Units that mean "data is stored in radians, display in degrees"
_RAD_TO_DEG_UNITS = {"deg", "deg/s", "deg/s^2"}


@dataclass
class Panel:
    """
    One subplot in the dashboard.

    kind="time": each channel is drawn against its own logged time base.
    kind="xy":   exactly two channels [x, y]
    """
    title: str
    channels: list
    unit: str = ""
    kind: str = "time"
    hlines: tuple = ()
    labels: dict = field(default_factory=dict)  # channel -> legend label override


class FlightLogger:
    def __init__(self):
        self._times = defaultdict(list)
        self._values = defaultdict(list)

    def log(self, t, **channels):
        """Log any number of named scalar signals at time t"""
        for name, value in channels.items():
            self._times[name].append(t)
            self._values[name].append(float(value))

    def series(self, name):
        """(times, values) arrays for one channel."""
        return np.array(self._times[name]), np.array(self._values[name])

    def array(self, name):
        """Values array for one channel."""
        return np.array(self._values[name])

    def plot(self, panels, cols=2, title="Flight dashboard"):
        rows = -(-len(panels) // cols)  # ceil
        fig, axs = plt.subplots(rows, cols, figsize=(6.5 * cols, 2.9 * rows))
        axs = np.atleast_1d(axs).ravel()
        fig.suptitle(title, fontsize=16, fontweight='bold')

        for ax, panel in zip(axs, panels):
            self._draw_panel(ax, panel)
        for ax in axs[len(panels):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def _draw_panel(self, ax, panel):
        to_display = np.degrees if panel.unit in _RAD_TO_DEG_UNITS else lambda v: v

        if panel.kind == "xy":
            x_name, y_name = panel.channels
            ax.plot(to_display(self.array(x_name)), to_display(self.array(y_name)), lw=2)
            ax.set_xlabel(f"{panel.labels.get(x_name, x_name)} ({panel.unit})")
            ax.set_ylabel(f"{panel.labels.get(y_name, y_name)} ({panel.unit})")
            ax.axis('equal')
        else:
            for name in panel.channels:
                t, v = self.series(name)
                ax.plot(t, to_display(v), label=panel.labels.get(name, name), lw=2)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(panel.unit)
            ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
            if len(panel.channels) > 1:
                ax.legend(loc='upper right')

        for y in panel.hlines:
            ax.axhline(y, color='gray', linestyle=':')
        ax.set_title(panel.title, loc='left')
        ax.grid(True, linestyle=':', alpha=0.6)
