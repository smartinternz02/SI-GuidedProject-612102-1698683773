import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dash import html


with open('pickles\model.pkl', 'rb') as kmeans_file:
    model = pickle.load(kmeans_file)

with open('pickles\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

rfm_data = pd.read_csv('Data/rfm.csv')
df = pd.read_csv('Data/Retail.csv',  encoding="ISO-8859-1")
X = pd.read_csv('Data/X.csv')

def process():
    df = pd.read_csv('Data/Retail.csv',  encoding="ISO-8859-1")
    df = df.dropna()
    df = df[df['InvoiceNo'].apply(lambda x: 'C' not in x)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'],format='%d-%m-%Y %H.%M')
    df.drop_duplicates(inplace = True)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    max_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    df['Recency'] = (max_date - df['InvoiceDate'])
    df['Recency'] = df['Recency'].dt.days
    return df



def rfm(df):
    rfm_data = df.groupby('CustomerID').agg({
    'Recency': 'min',
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).reset_index()

# Rename columns
    rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Display the RFM data
    return  rfm_data

def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

def segment(rfm_data):
    quantiles = rfm_data.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()
    rfm_data['R_Quartile'] = rfm_data['Recency'].apply(RClass, args=('Recency',quantiles,))
    rfm_data['F_Quartile'] = rfm_data['Frequency'].apply(FMClass, args=('Frequency',quantiles,))
    rfm_data['M_Quartile'] = rfm_data['Monetary'].apply(FMClass, args=('Monetary',quantiles,))
    rfm_data['RFMClass'] = rfm_data.R_Quartile.map(str) \
                            + rfm_data.F_Quartile.map(str) \
                            + rfm_data.M_Quartile.map(str)
    return rfm_data

def customer_type(row):
    if row['RFMClass'] == '111':
        return 'Top Tier'
    elif row['F_Quartile'] == 1:
        return 'Frequent Supporters'
    elif row['M_Quartile'] == 1:
        return 'High Spenders'
    elif row['RFMClass'] == '322':
        return 'Inactive But Potentially Returning'
    elif row['RFMClass'] == '444':
        return 'Inactive and Unlikely to Return'
    else:
        return 'Other'

    
def RFM_Segment(rfm_data):
    rfm_data['RFM_Segment'] = rfm_data.apply(customer_type, axis=1)
    return rfm_data


def customer_habit(df,rfm_data):
    data = df.groupby('CustomerID').agg({'StockCode': 'nunique',  # Number of unique products purchased
    'Description': lambda x: x.value_counts().idxmax()  # Most frequently purchased product
}).reset_index()
    data.columns = ['CustomerID', 'UniqueProducts', 'MostFrequentProduct']
    segmented = pd.merge(rfm_data, data, on='CustomerID', how='left')
    return segmented

segmented = customer_habit(df,rfm_data)

def stats(segmented):
    cluster_stats = segmented.groupby('RFM_Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()

    stats_html = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in cluster_stats.columns])] +
        # Body
        [html.Tr([html.Td(cluster_stats.iloc[i][col]) for col in cluster_stats.columns]) for i in range(len(cluster_stats))]
    )
    return stats_html


def profile(segmented):
    sample_profiles = segmented.groupby('RFM_Segment').apply(lambda x: x.sample(1)).reset_index(drop=True)
    print("Sample Customer Profiles:")
    print(sample_profiles[['CustomerID', 'RFM_Segment', 'Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'MostFrequentProduct']])


def top_product(segmented):
    top_products = segmented.groupby('RFM_Segment')['MostFrequentProduct'].value_counts().unstack().fillna(0)
    print("Top Products for Each Segment:")
    print(top_products)

def Predict_cluster(data):
    data = scaler.transform(data)
    predicted_cluster = model.predict(data)[0]


    if predicted_cluster == 0:
        print("Description: High Recency, Low Frequency, Low Monetary")
        print('''
These customers have high recency, meaning they haven't made a purchase recently.
They have low frequency, indicating infrequent purchases.
Their monetary value is also low, suggesting smaller spending.
              ''')
    elif predicted_cluster == 1:
        print("Description: Moderate Recency, Moderate Frequency, Moderate Monetary")
        print(''' 
These customers have moderate recency, indicating a moderate time since their last purchase.
They make purchases with a moderate frequency.
Their monetary value is also at a moderate level.
''')
    elif predicted_cluster == 2:
       print("Description: Low Recency, High Frequency, High Monetary")
       print(''' 
These customers have low recency, meaning they recently made a purchase.
They exhibit high frequency, suggesting frequent purchases.
Their monetary value is high, indicating higher spending.
''')
    else:
       print("Unknown Cluster")

def customer_d(id,df = process()):
    customer_data = df[df['CustomerID'] == id]
    return customer_data

def show_customer_details(customer_id, segmented):
    customer_data = segmented[segmented['CustomerID'] == customer_id]

    # Display customer details
    customer_details = customer_data[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Segment', 'UniqueProducts', 'MostFrequentProduct']]

    return customer_details
   
    # Display top products for the customer's segment
    top_products_customer_segment = df[df['RFM_Segment'] == customer_data['RFM_Segment'].iloc[0]]
    top_products = top_products_customer_segment.groupby('MostFrequentProduct')['CustomerID'].count().sort_values(ascending=False)[:5]
    print("\nTop Products for Customer's Segment:")
    print(top_products)

    # Display a pie chart of customer types
    customer_type_counts = segmented['RFM_Segment'].value_counts()
    fig_pie = px.pie(customer_type_counts, names=customer_type_counts.index, title='Customer Type Distribution')
    fig_pie.show()



def graph1(segmented):
    # Plotting Recency vs Frequency with color-coded clusters
    fig = px.scatter(segmented, x='Recency', y='Frequency', color='RFM_Segment',
                     size='Monetary', title='Customer Segmentation - Recency vs Frequency',
                     labels={'Recency': 'Recency', 'Frequency': 'Frequency', 'Monetary': 'Monetary'})
   
    return fig
#fig1 = graph1(rfm_data)
#fig1.show()

def graph2(segmented):
    # Plotting a bar chart of cluster sizes
    cluster_sizes = segmented['RFM_Segment'].value_counts().sort_index()
    fig = px.bar(x=cluster_sizes.index, y=cluster_sizes.values,
                 title='Cluster Sizes', labels={'x': 'RFM Segment', 'y': 'Number of Customers'})
    return fig
#fig2 = graph2(rfm_data)
#fig2.show()

def graph3(segmented):
    # Plotting a scatter plot of Recency vs Monetary with color-coded clusters
    fig = px.scatter(segmented, x='Recency', y='Monetary', color='RFM_Segment',
                     size='Frequency', title='Customer Segmentation - Recency vs Monetary',
                     labels={'Recency': 'Recency', 'Monetary': 'Monetary', 'Frequency': 'Frequency'})
    return fig

#fig3 = graph3(rfm_data)
#fig3.show()

def graph5(segmented):
    # Plotting a 3D scatter plot of Recency, Frequency, and Monetary with color-coded clusters
    fig = px.scatter_3d(segmented, x='Recency', y='Frequency', z='Monetary', color='RFM_Segment',
                        size='Frequency', title='Customer Segmentation - 3D Scatter Plot',
                        labels={'Recency': 'Recency', 'Frequency': 'Frequency', 'Monetary': 'Monetary'})
    return fig

#fig5 = graph5(rfm_data)
#fig5.show()

def graph6(segmented):
    # Plotting a bar chart of Unique Products for each customer segment
    fig = px.bar(segmented, x='RFM_Segment', y='UniqueProducts',
                 title='Number of Unique Products for Each Customer Segment',
                 labels={'RFM_Segment': 'Customer Segment', 'UniqueProducts': 'Number of Unique Products'})
    return fig
#fig6 = graph6(segmented)
#fig6.show()

import plotly.express as px

def pie_chart_customer_type(segmented):
    customer_type_counts = segmented['RFM_Segment'].value_counts().reset_index()
    customer_type_counts.columns = ['RFM_Segment', 'Count']
    
    fig = px.pie(customer_type_counts, names='RFM_Segment', values='Count',
                 title='Distribution of Customer Types',
                 labels={'RFM_Segment': 'Customer Type', 'Count': 'Count'})

    return fig

#fig7 = pie_chart_customer_type(rfm_data)
#fig7.show()
# To use this function, call it and show the plot:
# fig_pie = pie_chart_customer_type(segmented)
# fig_pie.show()
def scatter_plot_clusters(segmented):
    fig = px.scatter(segmented, x='Recency', y='Frequency', color='Cluster',
                     title='Customer Segmentation - Recency vs Frequency',
                     labels={'Recency': 'Recency', 'Frequency': 'Frequency', 'Cluster': 'Cluster'})
    return fig

#fig8 = scatter_plot_clusters(rfm_data)
#fig8.show()

def scatter_plot_clusters_fm(segmented):
    fig = px.scatter(segmented, x='Frequency', y='Monetary', color='Cluster',
                     title='Customer Segmentation - Frequency vs Monetary',
                     labels={'Frequency': 'Frequency', 'Monetary': 'Monetary', 'Cluster': 'Cluster'})
    return fig

#fig9 = scatter_plot_clusters_fm(rfm_data)
#fig9.show()

def cluster_size_bar_chart(X):
    cluster_sizes = X['Cluster'].value_counts().sort_index().reset_index()
    cluster_sizes.columns = ['Cluster', 'Count']
    
    fig = px.bar(cluster_sizes, x='Cluster', y='Count',
                 title='Cluster Sizes',
                 labels={'Cluster': 'Cluster', 'Count': 'Count'})
    
    return fig

#fig10 = cluster_size_bar_chart(rfm_data)
#fig10.show()


def time_series_plot(customer_data):
    fig = px.line(customer_data, x='InvoiceDate', y='TotalPrice', title='Time Series Plot of Purchases Over Time')
    return fig

def top_products_chart(customer_data, top_n=5):
    top_products = customer_data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(top_n)
    fig = px.bar(data_frame=top_products.reset_index(), x='Description', y='Quantity', labels={'Description': 'Product', 'Quantity': 'Quantity'},
                 title=f'Top {top_n} Products Purchased by the Customer')
    return fig


def purchase_frequency_histogram(customer_data):
    fig = px.histogram(customer_data, x='InvoiceDate', nbins=20, title='Purchase Frequency Distribution')
    return fig

def monetary_value_histogram(customer_data):
    fig = px.histogram(customer_data, x='TotalPrice', nbins=20, title='Monetary Value Distribution')
    return fig

def rfm_radar_chart(id):
    df = rfm(process())
    customer_data = df[df['CustomerID'] == id]

    fig = go.Figure()

    # Normalize the RFM values
    normalized_rfm = (customer_data[['Recency', 'Frequency', 'Monetary']] - customer_data[['Recency', 'Frequency', 'Monetary']].min()) / (customer_data[['Recency', 'Frequency', 'Monetary']].max() - customer_data[['Recency', 'Frequency', 'Monetary']].min())

    # Add radar chart traces
    fig.add_trace(go.Scatterpolar(
        r=normalized_rfm.iloc[0].values,
        theta=['Recency', 'Frequency', 'Monetary'],
        fill='toself',
        name='RFM Radar Chart'
    ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='RFM Radar Chart')
    
    return fig


def parse_rfm_input(r_input, f_input, m_input):
    try:
        # Convert inputs to integers
        r_value = int(r_input)
        f_value = int(f_input)
        m_value = int(m_input)

        # Return an array with RFM values
        return [r_value, f_value, m_value]
    except ValueError:
        # Handle the case where inputs are not valid integers
        return None


