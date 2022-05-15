from collections import deque
import dearpygui.dearpygui as dpg
from time import perf_counter

from nicepipe.utils.logging import steal_progress_bar_info

# TODO:
# pass longer description as arbitary field to progress bar so that it can appear
# here as a tooltip


def show_fps(len_duration=5):
    """len_duration is the total duration of data to keep in seconds"""
    data = {}
    series = {}
    x_data = deque([perf_counter()])
    with dpg.plot(label="Loop FPS", height=-1, width=-1, tag="fps_plot"):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_PlotPadding, 0, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LabelPadding, 0, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LegendPadding, 0, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LegendInnerPadding,
                    0,
                    1,
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_PlotBorderSize, 0, category=dpg.mvThemeCat_Plots
                )

        dpg.bind_item_theme("fps_plot", theme)

        dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)

        fps_x = dpg.add_plot_axis(
            dpg.mvXAxis, label="Time", no_gridlines=True, invert=True
        )
        fps_y = dpg.add_plot_axis(dpg.mvYAxis, label="FPS", no_tick_marks=True)
        dpg.set_axis_limits(fps_y, 0, 60)
        dpg.set_axis_ticks(fps_y, tuple((str(x), x) for x in range(10, 61, 10)))

    def update():
        """delta is time passed since last update in seconds"""

        x_data.append(perf_counter())
        while x_data[-1] - x_data[0] > len_duration:
            x_data.popleft()

        for task in steal_progress_bar_info():
            speed = 0.0 if task.speed is None else task.speed

            if not task.id in data:
                y_data = deque([0.0] * len(x_data))
                y_data[-1] = speed
                data[task.id] = y_data
                series[task.id] = dpg.add_line_series(
                    list(x_data), list(y_data), label=task.description, parent=fps_y
                )
            else:
                y_data = data[task.id]
                y_data.append(speed)

                while len(y_data) > len(x_data):
                    y_data.popleft()

                line = series[task.id]

                dpg.set_value(line, (list(x_data), list(y_data)))
                dpg.set_item_label(line, f"[{speed:.1f}Hz] {task.description}")

        dpg.fit_axis_data(fps_x)

    return update
