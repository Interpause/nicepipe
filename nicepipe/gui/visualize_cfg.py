from __future__ import annotations

from dearpygui import dearpygui as dpg


def attach_visualize_cfg(gui_sink, window_tag):
    def _config_toggle(checkbox, value, user_data):
        gui_sink.visuals_enabled[user_data] = value

    for viz in gui_sink.visualizers:
        dpg.add_checkbox(
            label=viz,
            user_data=viz,
            callback=_config_toggle,
            default_value=gui_sink.visuals_enabled.get(viz, False),
            parent=window_tag,
        )
