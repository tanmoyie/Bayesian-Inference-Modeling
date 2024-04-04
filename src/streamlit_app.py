# -*- coding: utf-8 -*-
""" Input: trained model, X_test[0] (from UI), build_feature.py, predict_model.py
Output: y_test[0]"""

import joblib
import pandas as pd
import numpy as np

import predict_model
import streamlit as st

# import model
model = joblib.load('../models/model_BIMReTA.pkl')


def rank_response_technologies(dispersion, E_ss, E_sl, E_sw, sufficient_mixing_energy,
                               E_ssC, seawater, E_ssI, soot_pollution, displacement):
    """Ranking oil spill response technologies in Arctic
    ++
    Motivated from https://github.com/krishnaik06/Dockers/blob/master/app.py
    """
    X1 = pd.DataFrame(np.array([[dispersion, E_ss, E_sl, E_sw, sufficient_mixing_energy,
                                 E_ssC, seawater, E_ssI, soot_pollution, displacement]]))
    # 99, 1, 1, 1, 'no', 0, 'Small', 1, 0, 'yes'
    X1.columns = ['evaporation_and_natural_disperson', 'E_ss', 'E_sl', 'E_sw','sufficient_mixing_energy',
                  'E_ssC', 'seawater', 'E_ssI', 'soot_pollution', 'displacement']

    prediction = model.predict(X1)
    print(prediction)
    return prediction


def main():
    st.title("rank_response_technologies")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Ranking spill response technology: ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        dispersion = st.number_input("Dispersion in %", min_value=10, max_value=99)  # a value of 10-99 unit?"
        sufficient_mixing_energy = st.selectbox("Sufficient mixing energy", ("yes", "no"))
        seawater = st.selectbox("Seawater volume", ("Small", "Large"))
        soot_pollution = st.selectbox("Soot pollution", ("YES soot pollution", "NO soot pollution"))
        displacement = st.selectbox("Displacement", ("yes", "no"))
    with col2:
        E_ss = st.radio("E_ss.mcr (Effect index of seasurface)", options=[-1, 0, 1], index=0)
        E_sl = st.selectbox('E_sl.mcr', options=[-1, 0, 1], index=0)
        E_sw = st.text_input("E_sw.mcr", options=[-1, 0, 1], index=0)
        E_ssC = st.text_input("E_ssC.cdu", options=[-1, 0, 1], index=1)
        E_ssI = st.text_input("E_ssI.isb", options=[-1, 0, 1], index=1)

    result = ""
    if st.button("Rank Technology"):
        result = rank_response_technologies(dispersion, E_ss, E_sl, E_sw, sufficient_mixing_energy,
                                            E_ssC, seawater, E_ssI, soot_pollution, displacement)
        result_df = pd.DataFrame(columns=["MCR", "CDU", "ISB"])
        result_df.index = ['Ranking']
        result_df.append(result)

        st.success(result_df) # need to tab one level up?

    if st.button("Publication"):
        st.text("Das, T., & Goerlandt, F. (2022). Bayesian inference modeling to rank response technologies in arctic marine oil spills. "
                "Marine Pollution Bulletin, 185, 114203")
        st.text("https://doi.org/10.1016/j.marpolbul.2022.114203")


if __name__ == '__main__':
    main()
