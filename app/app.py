import dash
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import datetime
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
from functions import *

# Load the model and data
with open('D:\projects\customer\SI-GuidedProject-612102-1698683773\model.pkl', 'rb') as kmeans_file:
    model = pickle.load(kmeans_file)

with open('D:\projects\customer\SI-GuidedProject-612102-1698683773\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

rfm_data = pd.read_csv('D:/projects/customer/SI-GuidedProject-612102-1698683773/Data/rfm.csv')
df = pd.read_csv('D:/projects/customer/SI-GuidedProject-612102-1698683773/Data/Retail.csv',  encoding="ISO-8859-1")

# Create a Dash web application
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.COSMO])


graph1_component = dcc.Graph(id='graph1')
graph2_component = dcc.Graph(id='graph2')
graph3_component = dcc.Graph(id='graph3')
graph4_component = dcc.Graph(id='graph4')


customer_details_tab_layout = html.Div([
    html.Label("Enter Customer ID:"),
    dcc.Input(id='customer-id-input', type='number', placeholder='Enter Customer ID'),
    html.Button('Show Details', id='show-details-button'),
    html.Div(id='customer-details-output'),
    html.Div([graph1_component, graph2_component, graph3_component, graph4_component])
])
prediction_layout = html.Div([
    html.Label("Enter RFM values "),
    dcc.Input(id='r-input', type='number', placeholder='Enter recency'),
    dcc.Input(id='f-input', type='number', placeholder='Enter frequency'),
    dcc.Input(id='m-input', type='number', placeholder='Enter monetary'),
    html.Button('Predict Cluster', id='predict-cluster-button'),
    html.Div(id='cluster-prediction-output')
])    


@app.callback(
    [Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Output('graph3', 'figure'),
    Output('graph4', 'figure')],
    [Input('show-details-button', 'n_clicks')],
    [State('customer-id-input', 'value')]
)
def update_customer_detail_graphs(n_clicks, customer_id):
    if n_clicks and customer_id:
        # Retrieve the customer data based on the customer_id
        loading_output = html.Div("Loading...")

        customer_data = customer_d(customer_id)  

        # Generate the figures for each graph based on customer data
        fig1 = time_series_plot(customer_data)
        fig2 = top_products_chart(customer_data)
        fig3 = purchase_frequency_histogram(customer_data)
        fig4 = monetary_value_histogram(customer_data)

        return fig1, fig2, fig3, fig4
    else:
        # Return empty figures if no data is available
        return {}, {}, {}, {}


@app.callback(Output('customer-details-output', 'children'), Input('show-details-button', 'n_clicks'), State('customer-id-input', 'value'))
def show_customer_details_callback(n_clicks, customer_id):
    if n_clicks and customer_id:
        customer_details = show_customer_details(customer_id, segmented)        
        customer_details_layout = html.Div([
            html.Label("Customer Details:"),
            html.Table([
                html.Tr([html.Th(col) for col in customer_details.columns]),
                html.Tr([html.Td(customer_details[col]) for col in customer_details.columns]),
            ])
        ])

        return customer_details_layout

    return None


@app.callback(Output('cluster-prediction-output', 'children'), Input('predict-cluster-button', 'n_clicks'), State[('r-input', 'value'),('f-input', 'value'),('m-input', 'value')])
def predict_cluster(n_clicks, r_input,f_input,m_input):
    if n_clicks and r_input and f_input and m_input:
        # Parse the RFM values from the input
        rfm_values = parse_rfm_input(r_input, f_input, m_input)  # Implement a function to parse RFM input

        if rfm_values is not None:
            # Predict the cluster based on the parsed RFM values
            predicted_cluster, description = Predict_cluster(rfm_values)

            # Create the layout to display cluster details
            cluster_details_layout = html.Div([
                html.Label("Predicted Cluster:"),
                html.P(f"Cluster {predicted_cluster}"),
                html.Label("Description:"),
                html.P(description)
            ])

            return cluster_details_layout
        else:
            return html.P("Invalid RFM input. Please use the format: R:X, F:Y, M:Z")

    return None


segment_cluster_dropdown = dcc.Dropdown(
    id='segment-cluster-dropdown',
    options=[
        {'label': 'Segments', 'value': 'segments'},
        {'label': 'Clusters', 'value': 'clusters'}
    ],
    value='segments'  # Default selection
)

# Define a callback to display the selected graph based on user selections
@app.callback(
    Output('selected-graph', 'children'),
    Input('segment-cluster-dropdown', 'value')
)
def display_selected_graph(selected_option):
    if selected_option == 'segments':
            selected_graph = [
            dcc.Graph(figure=graph1(rfm_data)),
            dcc.Graph(figure=graph2(rfm_data)),
            dcc.Graph(figure=graph3(rfm_data)),
            dcc.Graph(figure=graph5(rfm_data)),
            dcc.Graph(figure=graph6(segmented)),
            dcc.Graph(figure=pie_chart_customer_type(rfm_data)),
            stats(segmented)
        ]
    elif selected_option == 'clusters':
            selected_graph = [
            dcc.Graph(figure=scatter_plot_clusters(rfm_data)),
            dcc.Graph(figure=scatter_plot_clusters_fm(rfm_data)),
            dcc.Graph(figure=cluster_size_bar_chart(rfm_data))
        ]
    else:
        selected_graph = html.Div()  # Empty graph if no option is selected
    
    return selected_graph

# Layout for the Dash app
app.layout = html.Div([
    html.H1("Customer Segmentation Dashboard"),
    dcc.Tabs(id="tabs", value='general-segmentation', children=[
        dcc.Tab(label='General Segmentation', value='general-segmentation'),
        dcc.Tab(label='Customer Details', value='customer-details'),
        dcc.Tab(label='Cluster Prediction', value='cluster-prediction')
    ]),
    html.Div(id='tab-content'),
])

@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab_content(tab):
    if tab == 'general-segmentation':
        return html.Div([
            html.Label("Select Segments or Clusters:"),
            segment_cluster_dropdown,
            html.Div(id='selected-graph')
        ])

    elif tab == 'customer-details':
        return customer_details_tab_layout
    elif tab == 'cluster-prediction':
        return prediction_layout

if __name__ == '__main__':
    app.run_server(debug=True)
