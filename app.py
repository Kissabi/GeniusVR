import asyncio
import base64
import dash, cv2
from dash import dcc, html
from dash.exceptions import PreventUpdate
import threading
import dash_mantine_components as dmc
from dash import Output, Input, State, callback
import dash_daq as daq
from quart import Quart, websocket
from dash_extensions import WebSocket
from ultralytics import YOLO
import plotly.graph_objs as go
import plotly.express as px
from functools import partial
import random
from dash_iconify import DashIconify
from plotly.subplots import make_subplots
import json

model = YOLO('yolov8n.pt')

class VideoCamera(object):
    def __init__(self, video_path,classes=0,pred_cls=None):
        self.video = cv2.VideoCapture(video_path)
        self.classes = classes
        self.pred_cls = pred_cls
        
    def __del__(self):
        self.video.release()

    def get_frame(self, dta=False):
        success, frame = self.video.read()

        if success:
            results = model(frame,classes=self.classes)
            if dta is True:
                get_cls = []
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        get_cls.append(result.names[int(box.cls[0])])
                        print(f"foram detectados: {get_cls}")
                return get_cls
            else:
                annotated_frame = results[0].plot()
                
                _, jpeg = cv2.imencode('.jpg', annotated_frame)
                return jpeg.tobytes()
                
        return None




async def stream(camera, delay=None):
    while True:
        if delay is not None:
            await asyncio.sleep(delay)  # add delay if CPU usage is too high
        frame1 = camera.get_frame(dta=True)
        frame2 = camera.get_frame()
        print(frame1)
        await websocket.send(json.dumps(frame1))
        await websocket.send(f"data:image/jpeg;base64, {base64.b64encode(frame2).decode()}")
        
        
# Setup small Quart server for streaming via websocket, one for each stream.
server = Quart(__name__)
n_streams = 2      


        
@server.websocket("/stream0")
async def stream0():
    camera = VideoCamera("assets/example2.mp4",classes=39)
    await stream(camera)
    

@server.websocket("/stream1")
async def stream1():
    camera = VideoCamera("assets/example1.mp4")
    await stream(camera)


# Create Dash application for UI.
app = dash.Dash(__name__)

theme = {
    'dark': True,
    'color':'#FFFFFF',
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}
def graph(ent):
        fig = go.Figure()
        if ent is None:
                raise PreventUpdate
            
        if ent == 'Data':
            fig = make_subplots(rows=3, cols=1)
            
            fig.add_trace(go.Scatter(y = [1.4,7,2,4], x = [1,2,3,4], mode="lines", name = "G1", hovertext=["MWh","Hours in Day"]),row=1,col=1)

            
            fig.add_trace(go.Scatter(y = [1.5,3,6,4], x = [1,2,3,4], mode="lines", name = "G2", hovertext=["MWh","Hours in Day"]),row=2,col=1)

            
            fig.add_trace(go.Scatter(y = [1.2,6,2,2], x = [1,2,3,4], mode="lines", name = "G3", hovertext=["MWh","Hours in Day"]),row=3,col=1)

        
        if ent == "off":
            fig = px.line(x = [0], y = [0], title="TURN ON TO MONITOR POWER CONSUMPTION")
            fig.layout.yaxis.title = "MWh"
            fig.layout.xaxis.title = "Hours in Day"
          
        
        fig.layout.template='plotly_dark'
        fig.layout.font=dict(color='#fff')
        fig.layout.paper_bgcolor = '#333'
        fig.layout.plot_bgcolor = '#333'
    
        return fig
        
header = dmc.Grid([dmc.Col(html.Div([html.H3("REAL-TIME MANUFACTORING MONITORING CONCEPT")],style={"border": "solid 1px #A2B1C6","backgroundColor":"#333","margin-top":"20px","padding":"10px","border-radius":"20px", "textAlign":"center"}),span=7)],justify="center",align="center")
            
rootLayout = html.Div([
    dmc.Grid(
    children=[
    dmc.Col(html.Div([daq.StopButton(label=' ',id='darktheme-daq-led',className='dark-theme-control'),dmc.Space(h=200),html.Img(src='assets/glass.png', style={'width':'350px','height':'100px','textAlign':''})],style={"margin-top":"20px"}),span=3),
    dmc.Col(html.Div([dmc.Space(w=20),daq.LEDDisplay(label='OPERATOR ID',value=1126,color=theme['primary'],id='darktheme-daq-graduated',className='dark-theme-control')],style={"margin-top":"40px"}),span=6),
    dmc.Col(html.Div([daq.Knob(label='Air Pressure (bar)',value=6,color=theme['primary'],id='daq-knob',className='dark-theme-control')],style={"margin-top":"20px"}),span='auto'),
    ]),
])


metric = {
    "m0": 2,
    "m1": 5,
    "m2": 3
}

LayoutThermomether = html.Div([daq.LEDDisplay(label='Bottles',value=0,color=theme['primary'],id='daq-led0',className='dark-theme-control'),html.Br(),dmc.Space(h=10),daq.Thermometer(label='Temperature (ºC)',id='daq-thermometer',value=190, min=180, max=200, className='dark-theme-control'),dmc.Space(h=20)])
LayoutTank = html.Div([daq.LEDDisplay(label='People in Restrited Areas',value=0,color=theme['primary'],id='daq-led1',className='dark-theme-control'),dmc.Space(h=20),daq.Tank(label='Sand Tank (T)',id='daq-tank',color='#C2B280',value=10,className='dark-theme-control'),dmc.Space(h=30)])
LayoutKPI = html.Div([daq.ToggleSwitch(label=' ',className='dark-theme-control', color=theme['primary'],id="tog",value=False), dmc.Space(h=70),daq.GraduatedBar(id='our-graduated-bar0',label="Remaing Power (G1)",value=metric['m0'],color='blue',className='dark-theme-control'),html.Br(),daq.GraduatedBar(id='our-graduated-bar1',label="Remaing Power (G2)",value=metric['m1'],color='orange',className='dark-theme-control'),html.Br(),daq.GraduatedBar(id='our-graduated-bar2',label="Remaing Power (G3)",value=metric['m2'],color=theme['primary'],className='dark-theme-control'),dmc.Space(h=60)])

app.layout = dmc.NotificationsProvider(html.Div([
    html.Div(children=[header]),
    html.Div(id='dark-theme-components-1', children=[
        daq.DarkThemeProvider(theme=theme, children=rootLayout)
    ], style={
        'border': 'solid 1px #A2B1C6',
        'border-radius': '10px',
        'backgroundColor':'#303030',
        'color':'#FFFFFF',
        'margin-left':'20px',
        'margin-right':'20px',
        'margin-top':'20px'
        
    }),
    html.Br(),
    html.Div(id='tank-alert'),
    html.Div(id='daq-led1-alert'),
    html.Br(),
    dcc.Interval(id='interval-component', interval=10000, n_intervals=0),
    dcc.Interval(id='interval-component2', interval=1000000, n_intervals=0, max_intervals=0),
    html.Div([
    dmc.Grid(
    children=[
    dmc.Col(html.Div([daq.DarkThemeProvider(theme=theme, children=LayoutTank)],style={'border': 'solid 1px #A2B1C6','backgroundColor':'#333','border-radius': '10px','margin-left': '20px'}),span=3),
    dmc.Col(html.Div([
    html.Header('Camera 1',style={"margin-left":"80px","margin-top":"10px"}),
    html.Header('Camera 2',style={"margin-left":"300px","margin-top":"-20px"}),
    html.Div(
    [html.Img(style={'width': '40%','height':'300px','padding': 10}, id=f"v{i}",className='dark-theme-control') for i in range(n_streams)] +
    [WebSocket(url=f"ws://127.0.0.1:5000/stream{i}", id=f"ws{i}") for i in range(n_streams)],
    )],style={
        'border': 'solid 1px #A2B1C6',
        'backgroundColor':'#333',
        'color':'#FFFFFF',
        'border-radius': '10px'
        
    }),span=6),
    
    dmc.Col(html.Div([daq.DarkThemeProvider(theme=theme, children=LayoutThermomether)],style={'border': 'solid 1px #A2B1C6','backgroundColor':'#333','border-radius': '10px','margin-right': '20px'}),span=2),
    ],style={
        'border-radius': '5px',
        'margin-top': '20px',
        'color':'#FFFFFF'
    }),
]),
    

    html.Br(),
    html.Div([
    dmc.Grid(
    children=[
    dmc.Col(html.Div([dcc.Graph(id='kpi', figure = graph("Data"))],style={'border': 'solid 1px #A2B1C6','backgroundColor':'#333','border-radius': '10px','margin-left': '20px'}),span='auto'),
    dmc.Col(html.Div([daq.DarkThemeProvider(theme=theme, children=LayoutKPI)],style={'border': 'solid 1px #A2B1C6','backgroundColor':'#333','border-radius': '10px','padding':'30px','margin-right': '20px'}),span=4)
    ],),], style={
        'border-radius': '5px',
        'margin-top': '20px',
        'color':'#FFFFFF'
    }),
],style={
        'backgroundColor':'#595855',
        'color':'#FFFFFF'
    }))
    

@callback(
    Output('tog', 'color'),
    Output('kpi', 'figure'),
    Input('tog', 'value')
)

def update_output(value):
    if value is False:
        return "red",graph("off")
    else:
        return theme["primary"],graph("Data")



    

def update_output(i, value):
    if value is False:
        return 0
    else: 
        return metric[f"m{i}"]

for i in range(0, 3):
    callback_func = partial(update_output, i)
    app.callback(Output(f"our-graduated-bar{i}", "value"), Input("tog", "value"))(callback_func)
    


@app.callback(
    Output('daq-thermometer', 'value'),
    Input('interval-component', 'n_intervals')
    )

def update_random_value(value):
    return random.randint(190, 195)


@app.callback(
    Output('daq-tank', 'value'),
    Input('interval-component', 'n_intervals'),
    State('daq-tank','value')
    )

def update_random_value(value1,value2):
    return value2-1






@app.callback(
    Output('tank-alert','children'),
    Input('interval-component', 'n_intervals'),
    Input('daq-tank','value'),
    prevent_initial_call = False
    )

def update_random_value(value1,value2):
    if value2 == 0:
        return dmc.Notification(
            title="Warning",
            id="simple-notify1",
            action="show",
            message="The Sand Tank Has Run Out!",
            autoClose=20000,
            icon=DashIconify(icon="carbon:warning-hex-filled", color='red'),
        )
           
        
    else:
         return ''


@app.callback(
    Output('daq-led1-alert','children'),
    Output('daq-led1','color'),
    Input('interval-component', 'n_intervals'),
    Input('daq-led1','value'),
    prevent_initial_call = False
    )


def update_random_value(value1,value2):
    if value2 < 1:
        raise PreventUpdate
    
    else:
        notif = dmc.Notification(
        title="Warning",
        id="simple-notify2",
        action="show",
        message="A Person Sighted in a Restricted Area!",
        autoClose=20000000,
        icon=DashIconify(icon="mdi:person-warning",color='red'),
        )
        color = "red"
        return notif, color
    
      



def update_output(value1,value2,value3): 
    if len(value1['data'])>50:
        return value3,value1['data']
    if ('bottle' in value1['data'] or 'person' in value1['data']) and len(value1['data']) < 100:
        return int(len(value1['data'])/10),value2
    
        
for i in range(2):
    callback_func = partial(update_output)
    app.callback(Output(f"daq-led{i}","value"),Output(f"v{i}","src"),Input(f"ws{i}", "message"),State(f"v{i}","src"),State(f"daq-led{i}","value"))(callback_func)






if __name__ == '__main__':
    threading.Thread(target=app.run_server).start()
    server.run()