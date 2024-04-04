# -*- coding: utf-8 -*-
""" Input: trained model, X_test[0] (from UI), build_feature.py, predict_model.py
Output: y_test[0]"""

import joblib
import pandas as pd
import numpy as np

# import predict_model
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

    prediction = model.predict(X1).reshape(1,-1).flatten().tolist()
    return prediction


def main():
    # global result_df

    html_temp = """<div style="background-color:tomato;padding:10px;height=20px">
    <h3 style="color:white;text-align:center;">Ranking spill response technology: ML App </h3>
    </div>"""
    st.markdown(html_temp,  unsafe_allow_html=True)

    input_title = """<div style="background-color:gray;padding:0px">
    <h4 style="color:white;text-align:center;">Input variables </h4>
    </div>"""
    st.markdown(input_title, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        dispersion = st.number_input("Dispersion in %", min_value=10, max_value=99)  # a value of 10-99 unit?"
        sufficient_mixing_energy = st.selectbox("Sufficient mixing energy", ("yes", "no"), index=1)
        seawater = st.selectbox("Seawater volume", ("Small", "Large"))
        soot_pollution = st.selectbox("Soot pollution", ("YES soot pollution", "NO soot pollution"))

    with col2:
        displacement = st.selectbox("Displacement", ("yes", "no"), index=1)
        E_ss = st.radio("E_ss.mcr (Effect index of seasurface)", options=[-1, 0, 1], index=0)
        E_sl = st.radio('E_sl.mcr', options=[-1, 0, 1], index=0)
    with col3:
        E_sw = st.radio("E_sw.mcr", options=[-1, 0, 1], index=0)
        E_ssC = st.radio("E_ssC.cdu", options=[-1, 0, 1], index=1)
        E_ssI = st.radio("E_ssI.isb", options=[-1, 0, 1], index=1)

    output_title = """<div style="background-color:grey;padding:0px">
    <h4 style="color:white;text-align:center;"> Result </h4>
    </div>"""
    st.markdown(output_title, unsafe_allow_html=True)

    result = ""
    if st.button("Rank Technology"):
        result = rank_response_technologies(dispersion, E_ss, E_sl, E_sw, sufficient_mixing_energy,
                                            E_ssC, seawater, E_ssI, soot_pollution, displacement)

        result_df1 = pd.DataFrame(columns=["MCR", "CDU", "ISB"])
        #result_df = pd.concat([result_df1, pd.DataFrame(result)], ignore_index=True, axis=0)
        result_df1.loc[len(result_df1.index)] = result
        result_df1.set_index([['Ranking'] * len(result_df1)],
                             inplace=True)  # df.set_index([['A']*len(df)], inplace=True)
        st.success(st.dataframe(result_df1))  # need to tab one level up?

    paper_title = """<div style="background-color:None;padding:10px">
    <p style="color:None;text-align:left;"> Reference: Das, T., & Goerlandt, F. (2022). Bayesian inference modeling to rank response technologies in arctic marine oil spills. "
                "Marine Pollution Bulletin, 185, 114203 <a href="https://doi.org/10.1016/j.marpolbul.2022.114203"> (link) </a> </p>
    </div>"""
    st.markdown(paper_title, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
