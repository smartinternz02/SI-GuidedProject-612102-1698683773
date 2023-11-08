import dash
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
from functions import *

# Load the model and data
with open('pickles\model.pkl', 'rb') as kmeans_file:
    model = pickle.load(kmeans_file)

with open('pickles\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

rfm_data = pd.read_csv('Data/rfm.csv')
df = pd.read_csv('Data/Retail.csv',  encoding="ISO-8859-1")
X = pd.read_csv('Data/X.csv')
# Create a Dash web application
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.COSMO],suppress_callback_exceptions=True)
server = app.server


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


@app.callback(Output('cluster-prediction-output', 'children'), Input('predict-cluster-button', 'n_clicks'), [State('r-input', 'value'), State('f-input', 'value'), State('m-input', 'value')])
def predict_cluster(n_clicks, r_input, f_input, m_input):
    #print(f"n_clicks: {n_clicks}, r_input: {r_input}, f_input: {f_input}, m_input: {m_input}")

    if n_clicks and r_input is not None and f_input is not None and m_input is not None:
        # Combine the input RFM values
        try:
            rfm_values = [float(r_input), float(f_input), float(m_input)]
        except ValueError:
            return html.P("Invalid input. Please enter valid numeric RFM values.")

        # Display the user input
        user_input_layout = html.Div([
            html.Label("User Input RFM Values:"),
            html.P(f"Recency: {rfm_values[0]}, Frequency: {rfm_values[1]}, Monetary: {rfm_values[2]}")
        ])

        # Combine the input RFM values
        input_data = pd.DataFrame({'Recency': [rfm_values[0]], 'Frequency': [rfm_values[1]], 'Monetary': [rfm_values[2]]})

        #print("Input data:")
        #print(input_data)
        B = X[['Recency', 'Frequency', 'Monetary']]
        # Append input_data to rfm_data
# Append input_data to rfm_data
        combined_data = pd.concat([B,input_data], ignore_index=True)
        #print("Combined data:")
        #combined_data = combined_data[['Recency', 'Frequency', 'Monetary']]
        #print(combined_data.tail())

        # Standardize the combined data using the same scaler
        combined_data_scaled = scaler.transform(combined_data)

        #print(combined_data_scaled)
        # Predict the cluster for the input RFM values
        predicted_cluster = model.predict(combined_data_scaled)[-1]
        #print(f"Predicted cluster: {predicted_cluster}")

        
        # Create the layout to display the predicted cluster
        cluster_details_layout = html.Div([
            html.Label("Predicted Cluster:"),
            html.P(f"Cluster {predicted_cluster}")
        ])

        if predicted_cluster == 0:
            description = "High Recency, Low Frequency, Low Monetary. "
            additional_description = '''
            These customers have high recency, meaning they haven't made a purchase recently.
            They have low frequency, indicating infrequent purchases.
            Their monetary value is also low, suggesting smaller spending.
            '''
        elif predicted_cluster == 1:
            description = "Moderate Recency, Moderate Frequency, Moderate Monetary. "
            additional_description = '''
            These customers have moderate recency, indicating a moderate time since their last purchase.
            They make purchases with a moderate frequency.
            Their monetary value is also at a moderate level.
            '''
        elif predicted_cluster == 2:
            description = "Low Recency, High Frequency, High Monetary. "
            additional_description = '''
            These customers have low recency, meaning they recently made a purchase.
            They exhibit high frequency, suggesting frequent purchases.
            Their monetary value is high, indicating higher spending.
            '''
        else:
            description = "Unknown Cluster"
            additional_description = "No description available."

        cluster_description_layout = html.Div([
            html.Label("Cluster Description:"),
            html.P(description),
            html.P(additional_description)
        ])


        return [user_input_layout, cluster_details_layout, cluster_description_layout]
    else:
        return html.P("Invalid input. Please enter valid RFM values.")

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
