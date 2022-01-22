# Run this app with `python app.py` and
# visit http://127.0.0.1:1111/ in your web browser.

# import required packages
import os

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from dash import Output, Input
from statsmodels.tsa.stattools import pacf
import plotly.graph_objects as go 
import datetime
from datetime import date
from statsmodels.tsa.arima.model import ARIMA
from collections import OrderedDict

app = dash.Dash(__name__)
server = app.server
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

##
def load_ferry(path,route_number):
    '''
    Input:
    path: Ferry Data file path;
    route_number: Route Number from one place to another. E.g. FerryDH means ferry from Dartmouth to Halifax
    
    Output:
    a dataframe
    
    '''
    ferry = pd.read_csv(path)
    # transform to date format
    ferry['Route_Date'] = pd.to_datetime(ferry['Route_Date'])
    
    # select a specific route for future analysis
    Ferry=ferry[ferry['Route_Number'] == route_number]
    Ferry.index = Ferry['Route_Date']
    Ferry = Ferry.sort_index()
    Ferry = Ferry.drop('Route_Date',axis = 1)
    
    # impute missing value using pad method
    Ferry['Ridership_Total']=Ferry['Ridership_Total'].fillna(method='ffill')
    
    return Ferry

def get_ferry_data(Ferry,time_window):
    '''
    Inputs:
    Ferry: dataframe from load_ferry function
    route_number: Route Number from one place to another. E.g. FerryDH means ferry from Dartmouth to Halifax
    time_window: a list containing two time points. E.g. ['2017-01-01','2021-02-03']
    
    Outputs:
    Three plots and two dataframes (One is daily ridership, one is weekly average ridership)
    '''
    
    Ferry_slice = Ferry[np.logical_and(time_window[0] < Ferry.index, Ferry.index < time_window[1])]
    # Ferry_slice['Ridership_Total'].plot(figsize=(20,10),fontsize=20)
    # plt.title('Ferry Ridership of '+ route_number +' Per Day'+' from ' + time_window[0] +' to '+time_window[1], fontsize=20)
    # plt.show()
    
    Ferry_slice_week=pd.DataFrame(Ferry_slice.groupby(['Week_Range'])['Ridership_Total'].mean())
    # Ferry_slice_week['Ridership_Total'].plot(figsize=(20,10),fontsize=10)
    # plt.title('Ferry Ridership (on a weekly basis)',fontsize=20)
    # plt.show()
    
    
    #autocorrelation 
    # fig = plt.figure(figsize=(20,10))
    # pd.plotting.autocorrelation_plot(Ferry_slice_week['Ridership_Total'])
    # plt.title('Autocorrelation of Ferry Ridership (on a weekly basis)',fontsize=20)
    # plt.show()
    
    return Ferry_slice, Ferry_slice_week

def model_predict(Ferry, steps):
    '''
    Inputs:
    Ferry: Ferry dataframe
    Steps: how many days/weeks want to forecast (days or weeks depend on your Ferry dataframe based on which measure)
    
    Outputs:
    predictions for future ferry
    '''
    X = Ferry['Ridership_Total'].values
    history = [x for x in X]
    model = ARIMA(history, order=(8, 0, 3))
    model_fit = model.fit()
    output = model_fit.forecast(steps=steps)
    return np.round(output,0)

# see https://plotly.com/python/px-arguments/ for more options
app.title = 'Halifax Ferry Ridership Analytics'
app.layout = html.Div(
    children=[
        html.Div(
                children = [html.P(
                    children='🚢',
                    style={
                        'font-size': '48px',
                        'margin': '0 auto',
                        'text-align': 'center'
                    }),
                html.H1(
                    children='Ferry Ridership Forecasting Dashboard',
                    style={
                        'color': '#FFFFFF',
                        'font-size': '48px',
                        'font-weight': 'bold',
                        'text-align': 'center',
                        'margin': '0 auto',
                        'textAlign': 'center',
                        'font-family':'Arial'
                    }
                ),
                html.P(
                    children='''
                    An automated tool for Halifax ferry ridership trend analysis and forecasting
                ''',
                    style={
                        'color': '#CFCFCF',
                        'margin': '4px auto',
                        'max-width': '384px',
                        'textAlign': 'center',
                        'font-family':'Arial'
                    }
                )],
                style = {
                    'background-color': '#222222',
                    'height': '256px',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'center'
                }
        ),


    html.H4(
        'Select Ferry Route and Time Period to See Trend',
        style={
            'font-family':'Arial'
        }
    ),
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            dict(label=x, value=x) for x in ['FerryDH','FerryHD','FerryWH','FerryHW']
        ],
        value = 'FerryDH',
        placeholder="Select a ferry route",
        style={
            'font-family':'Arial'
        }
    ),
     dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=date(2017, 1, 1),
        max_date_allowed=date(2021, 10, 29),
        initial_visible_month=date(2017, 1, 1),
        start_date = date(2021,3,1),
        end_date=date(2021, 10, 29)
    ),

    dcc.Graph(
        id='daily-ridership-graph',
        style={
            'margin-bottom': '24px',
            'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)',
            'margin-right': 'auto',
            'margin-left': 'auto',
            'max-width': '1048px',
            'padding-right': '8px',
            'padding-left': '8px',
            'margin-top': '32px'
        }
    ),
    dcc.Graph(
        id='weekly-average-ridership-graph',
        style={
            'margin-bottom': '24px',
            'box-shadow': '0 4px 6px 0 rgba(0, 0, 0, 0.18)',
            'margin-right': 'auto',
            'margin-left': 'auto',
            'max-width': '1048px',
            'padding-right': '8px',
            'padding-left': '8px',
            'margin-top': '32px'
        }
    ),
    html.Div([
        html.H3(["Predict Ferry Ridership for How Many Days in Advance: ",
        dcc.Input(id='my-input-steps',type='number',value=7)],
        style={
            'textAlign': 'center',
            'margin':'auto',
            'font-family':'Arial'
        }), 

        html.Br(),

        html.Table(id = 'prediction-table',
        style={
            'textAlign': 'center',
            'margin':'auto',
            'font-family':'Arial'
        })
    ])
], style = {
        'font-family': "Lato",
        'margin': '0',
        'padding':'0',
        'background-color': '#F7F7F7'
})
@app.callback(
    Output('daily-ridership-graph', 'figure'),
    Output('weekly-average-ridership-graph', 'figure'),
    Output('prediction-table','children'),
    Input('demo-dropdown', 'value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input('my-input-steps', 'value')

)

def update_output(route_value,start_date,end_date,steps_value):
    Ferry = load_ferry(path = 'Transit_Ferry_Passenger_Counts.csv',route_number = route_value)
    Ferry_slice, Ferry_slice_week = get_ferry_data(Ferry, time_window = [start_date,end_date])
    fig1 = px.line(Ferry_slice, x=Ferry_slice.index, y=Ferry_slice['Ridership_Total'], title = 'Daily Ridership Trend',
                        labels={
                            'Route_Date':'',
                            'Ridership_Total':''
                        })
    fig2 = px.line(Ferry_slice_week, x=Ferry_slice_week.index, y=Ferry_slice_week['Ridership_Total'],title='Weekly Average Ridership Trend',
                        labels={
                                'Week_Range':'',
                                'Ridership_Total':''
                            })
    fig2.update_xaxes(showticklabels=False)
    #ACF Plot
    # fig3 = go.Figure()
    # fig3.add_trace(go.Scatter(
    # x= np.arange(len(pacf(Ferry_slice_week['Ridership_Total']))),
    # y= pacf(Ferry_slice_week['Ridership_Total'])
    # ))
 
    # pd.plotting.autocorrelation_plot(Ferry_slice_week['Ridership_Total'])
    future_preds=model_predict(Ferry_slice, steps = steps_value)
    end_date_datetime = date(int(end_date.split('-')[0]),int(end_date.split('-')[1]),int(end_date.split('-')[2]))
    data = OrderedDict(
    [
        ("Date", [str(end_date_datetime + datetime.timedelta(days=i+1)) for i in range(steps_value)]),
        ("Predicted Ridership", future_preds)
    ]
)
    df = pd.DataFrame(
    OrderedDict([(name, col_data) for (name, col_data) in data.items()])
)
    table = dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    page_action='none',
    style_table={'height': '200px', 'overflowY': 'auto'},
     style_header={
        'textAlign':'center',
        # 'backgroundColor': 'rgb(30, 30, 30)',
        # 'color': 'white'
    },
    style_data={
        'textAlign':'center',
        'width':'500px',
        # 'backgroundColor': 'rgb(50, 50, 50)',
        # 'color': 'white'
    },
)
    return fig1,fig2,table

if __name__ == '__main__':
    app.run_server(port=1111,debug=True)