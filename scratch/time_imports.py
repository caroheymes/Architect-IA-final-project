import time


def time_import(module_name):
    t0 = time.time()
    try:
        __import__(module_name)
        print(f"Imported {module_name} in {time.time() - t0:.3f}s")
    except Exception as e:
        print(f"Failed to import {module_name} in {time.time() - t0:.3f}s: {e}")


modules = [
    "os",
    "numpy",
    "pandas",
    "streamlit",
    "plotly.graph_objects",
    "plotly.express",
    "shapely.wkt",
    "pydeck",
    "json",
    "streamlit.components.v1",
]

for m in modules:
    time_import(m)
