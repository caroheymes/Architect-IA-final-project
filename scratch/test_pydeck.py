import time

t0 = time.time()
import pydeck as pdk

print(f"Imported pydeck in {time.time() - t0:.3f}s")
t1 = time.time()
layer = pdk.Layer("PathLayer", [])
try:
    r = pdk.Deck(layers=[layer]).to_html(open_browser=False)
    print(f"Rendered empty pydeck to html in {time.time() - t1:.3f}s")
except Exception as e:
    print(f"Error rendering pydeck: {e}")
