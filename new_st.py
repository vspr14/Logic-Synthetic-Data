import base64
import streamlit as st
import pandas as pd
import warnings
from tran_item_class import TranItem
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

warnings.filterwarnings("ignore")

def get_download_link(df, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="{text}.csv">Download {text}</a>'
    return href

def validate_number_input(number, min_value, max_value):
    if number < min_value:
        st.warning(f"Number must be greater than {min_value}.")
        return False
    elif number > max_value:
        st.warning(f"Number must be less than {max_value}.")
        return False
    else:
        return True

def reset_train():
    st.session_state.trained = False
    st.session_state.model.reset_metadata()

def change_model():
    st.session_state.model = model 

st.set_page_config(
   page_title="Synthetic Data Generation",
   layout="wide",
   initial_sidebar_state="expanded",
)

model = TranItem()

st.title("Item Sales Synthetic Data Generation")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if 'trained' not in st.session_state:
    st.session_state['trained'] = False

if 'model' not in st.session_state:
        st.session_state.model = model

if uploaded_file is not None:
    uploaded_df = pd.read_excel(uploaded_file)

    st.write("Preview of the uploaded data:")
    st.dataframe(uploaded_df.head())
    st.write("Total number of rows in dataframe: " + str(uploaded_df.shape[0]))
    st.write("Total number of columns in dataframe: " + str(uploaded_df.shape[1]))

    store_number = st.selectbox("Store Number", ["Choose an option"] + ["Any"] + uploaded_df["STORE"].unique().tolist(), key='store_number', on_change=reset_train)

    if st.session_state.store_number != "Choose an option":
        filtered_df = uploaded_df if store_number == "Any" else uploaded_df[uploaded_df["STORE"] == st.session_state.store_number]

        add_noise = st.radio("Add noise?", ["Yes", "No"], horizontal=True, index=None, on_change=reset_train, key='noise_radio')

        max_number = len(filtered_df) * 3
        min_number = 2
        number = st.number_input(f"Enter number of rows ({min_number} - {max_number}):", min_value=min_number, max_value=max_number)

        if st.button("Train"):
            if not st.session_state.trained:
                if add_noise in ["Yes", "No"]:
                        if validate_number_input(number, min_number, max_number):
                            st.session_state.model.set_df(filtered_df, uploaded_file, constraints=(not add_noise))
                st.session_state.trained = True

        if st.session_state.trained:
            if add_noise == "Yes":
                st.slider("Select the ratio of valid to invalid rows:", 0.1, 1.0, key='ratio1')
            if st.button("Synthesize Data"):
                
                if add_noise == "Yes":
                    result = st.session_state.model.gen_sample(number, st.session_state.noise_radio, st.session_state.ratio1)
                else:
                    result = st.session_state.model.gen_sample(number, False)
                st.write("Synthesized Data:")
                st.write(result.head())
                st.markdown(get_download_link(result, "Synthetic Data"), unsafe_allow_html=True)
                with st.spinner("Loading metrics..."):
                    metrics = st.session_state.model.metrics()
                    st.write(metrics[0])
                    st.dataframe(metrics[1])
                    st.write("### Visualizations")

                    min_rows = min(len(filtered_df), len(result))
                    sampled_filtered_df = st.session_state.model.df.sample(n=min_rows)
                    sampled_result = result.sample(n=min_rows)

                    qty_counts = sampled_filtered_df['QTY'].value_counts().sort_index()
                    fig_scatter_qty_original = px.scatter(
                        y=qty_counts.index, 
                        x=qty_counts.values, 
                        labels={'y': 'Original QTY', 'x': 'Frequency'},
                        title="Original QTY"
                    )

                    qty_counts_synth = sampled_result['QTY'].value_counts().sort_index()
                    fig_scatter_qty_synthetic = px.scatter(
                        y=qty_counts_synth.index, 
                        x=qty_counts_synth.values, 
                        labels={'y': 'Synthetic QTY', 'x': 'Frequency'},
                        title="Synthetic QTY"
                    )

                    # Display plots side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_scatter_qty_original)
                    with col2:
                        st.plotly_chart(fig_scatter_qty_synthetic)

                    unit_retail_counts = sampled_filtered_df['UNIT_RETAIL'].value_counts().sort_index()
                    fig_scatter_unit_retail_original = px.scatter(
                        y=unit_retail_counts.index, 
                        x=unit_retail_counts.values, 
                        labels={'y': 'Original UNIT_RETAIL', 'x': 'Frequency'},
                        title="Original UNIT_RETAIL"
                    )

                    unit_retail_counts_synth = sampled_result['UNIT_RETAIL'].value_counts().sort_index()
                    fig_scatter_unit_retail_counts_synthetic = px.scatter(
                        y=unit_retail_counts_synth.index, 
                        x=unit_retail_counts_synth.values, 
                        labels={'y': 'Synthetic UNIT_RETAIL', 'x': 'Frequency'},
                        title="Synthetic UNIT_RETAIL"
                    )

                    # Display plots side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_scatter_unit_retail_original)
                    with col2:
                        st.plotly_chart(fig_scatter_unit_retail_counts_synthetic)

                    st.write("#### kNN Visualization")
                    X = sampled_filtered_df[['QTY', 'UNIT_RETAIL']].values
                    y = np.zeros(X.shape[0])
                    X_synth = sampled_result[['QTY', 'UNIT_RETAIL']].values
                    y_synth = np.ones(X_synth.shape[0])
                    X_combined = np.vstack((X, X_synth))
                    y_combined = np.hstack((y, y_synth))

                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_combined, y_combined)
                    xx, yy = np.meshgrid(np.linspace(X_combined[:, 0].min(), X_combined[:, 0].max(), 100),
                                        np.linspace(X_combined[:, 1].min(), X_combined[:, 1].max(), 100))
                    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig_knn = go.Figure(data=[
                        go.Contour(x=xx[0], y=yy[:, 0], z=Z, showscale=False, colorscale='RdBu', opacity=0.4),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', name='Original Data', marker=dict(color='blue')),
                        go.Scatter(x=X_synth[:, 0], y=X_synth[:, 1], mode='markers', name='Synthetic Data', marker=dict(color='red'))
                    ])
                    st.plotly_chart(fig_knn)
                    reset_train()