import streamlit as st
import pydeck as pdk
import time

st.write("Hello pydeck starting")
t0 = time.time()
layer = pdk.Layer("PathLayer", [])
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=45.764043, longitude=4.835659, zoom=11.5),
        map_style="light",
    )
)
st.write(f"Hello pydeck completed in {time.time() - t0:.3f}s")
