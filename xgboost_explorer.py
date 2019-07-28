import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly import tools

import numpy as np
from scipy.special import expit
import pandas as pd

def mse(y,yhat):
    l = np.sum((y-yhat)**2) if len(y) > 0 else 0
    g = np.sum(y-yhat) if len(y) > 0 else np.nan
    h = np.sum(np.ones(len(y))) if len(y) > 0 else np.nan
    return {'loss':l,'grad':g,'hess':h}
    
    
def logloss(y,yhat):
    l = np.sum(-(y*np.log(yhat)+(1-y)*np.log(1-yhat))) if len(y) > 0 else np.nan
    g = np.sum(y-yhat) if len(y) > 0 else np.nan
    h = np.sum(yhat*(1-yhat)) if len(y) > 0 else np.nan
    return {'loss':l,'grad':g,'hess':h}


class Booster(object):
    def __init__(self,data,settings):#initialize the booster object
        self.settings = dict((k, v) for k, v in settings.items())
        self.cant_fit = False
        self.trxn_func = self.settings['transform'] or (lambda x: x)
        self.tree = self.__split(None,data.copy(deep=True),1)
        self.data=data
        if self.cant_fit:
            self.yhat = data['yhat']
        else:
            self.yhat = data['yhat']+self.predict(data['x'])
        
    def __get_opt(self,df):
        for i in df.index:
            left = self.settings['loss'](df.loc[:i-1,'y'],self.__transform(df.loc[:i-1,'yhat']))#get the loss, gradient, and hessian of left leaf
            right = self.settings['loss'](df.loc[i:,'y'],self.__transform(df.loc[i:,'yhat'])) #get the loss, gradient, and hessian of the right leaf
            df.loc[i,'loss_left'] = left['loss']
            df.loc[i,'loss_right'] = right['loss']
            df.loc[i,'grad_left'] = left['grad']
            df.loc[i,'grad_right'] = right['grad']
            df.loc[i,'hess_left'] = left['hess']
            df.loc[i,'hess_right'] = right['hess'] 
        df['min_child_weight'] = df[['hess_left','hess_right']].min(axis=1)
        
        #min child weight
        df['loss'] = np.where(df['min_child_weight'] >= self.settings['min_child_weight'],
                              -df['grad_left']**2/(df['hess_left']+self.settings['l2'])
                              -df['grad_right']**2/(df['hess_right']+self.settings['l2']),np.nan)
        
        #min split_loss
        df['loss'] = np.where(df['loss']< -self.settings['min_split_loss'],df['loss'],np.nan)
        
        df['weight_left'] = df['grad_left']/(df['hess_left']+self.settings['l2'])
        df['weight_right'] = df['grad_right']/(df['hess_right']+self.settings['l2'])
        return df
    
    def __split(self, tree, df, depth):
        df = df.copy(deep=True)
        tree = tree or {'left':None,'right':None}
        if (depth > self.settings['max_depth']) or (len(set(df['y'])) == 1):#add check for min child weight
            return None
        df = self.__get_opt(df.copy())
        if df['loss'].isnull().all():
            self.cant_fit = True
            return tree
        cut_ind = df['loss'].idxmin() #get the index of the best split
        cut = df.loc[cut_ind] #get the row for the best split
        tree['cut'] = (df['x'][cut_ind]+df['x'][cut_ind-1])/2 # get the x midpoint for the split
        tree['left'] = self.__split(tree['left'],df.loc[:cut_ind-1,:].copy(deep=True),depth+1) or cut['weight_left'] #continue to split
        tree['right'] = self.__split(tree['right'],df.loc[cut_ind:,:].copy(deep=True),depth+1) or cut['weight_right'] #continue to split
        tree['error'] = cut['loss']
        tree['opt'] = df
        return tree

        
    def __transform(self,values):
        return self.trxn_func(values)

    def predict(self,x):
        if self.cant_fit:
            return 0
        else:
            return np.array([self.__get_leaf(self.tree,x[i]) for i in range(len(x))])
            
    def __get_leaf(self,tree,x):
        if not isinstance(tree,dict): #return if leaf, else continue down branch
            return tree
        return self.__get_leaf(tree['left'],x) if x < tree['cut'] else self.__get_leaf(tree['right'],x)
    
    def get_loss(self):
        return self.__get_loss_recurse(self.tree)
    
    def __get_loss_recurse(self,tree):
        opt=pd.DataFrame()
        opt_df=tree['opt']
        if not isinstance(tree['left'],dict): #return if leaf, else continue down branch
            opt=opt.append(tree['opt'].loc[:opt_df['loss'].idxmin()-1])
        else:
            opt=opt.append(self.__get_loss_recurse(tree['left']))
            
        if not isinstance(tree['right'],dict): #return if leaf, else continue down branch
            opt=opt.append(tree['opt'].loc[opt_df['loss'].idxmin():])
        else:
            opt=opt.append(self.__get_loss_recurse(tree['right']))
        return opt

    

    
    ### create data
def random_draw(x):
    return np.random.binomial(1,x)


def true_function(x,func):
    if func == 'line':
        return (x)/10
    elif func == 'sine':
        return (np.sin(np.pi* x/4)+1)/2
    elif func == 'saw':
        return np.mod(x,6)/6
    elif func == 'step':
        return np.floor(x/1.25)/8

def create_noisy_data(func,noise,start,end,n,seed=0):
    np.random.seed(seed)
    x = np.linspace(start+(end-start)/(n+1),(end-start)*n/(n+1),n)
    true= true_function(x,func)
    y = true+np.random.randn(n)*noise/75
    return (x,y,true)

def transform(x):
    if settings['transform']==None:
        return x
    elif settings['transform']==expit:
        return expit(x)
    

def generate_data(problem,function, noise,n):
    df_d=pd.DataFrame()
    global seed
    df_d['x'],df_d['y'],df_d['true']=create_noisy_data(function,noise,0,10,n,seed)
    df_d['yhat']=np.mean(df_d['y'])
    df_d['preds']=df_d['yhat']
    if problem=='classification':
        df_d['y']=(df_d['y']-0.5)*4
        df_d['true']=(df_d['true']-0.5)*4
        df_d['true']=expit(df_d['true'])
        df_d['yhat']=np.mean(df_d['y'])
        df_d['y'] = random_draw(expit(df_d['y']))
        df_d['preds'] = expit(df_d['yhat'])
    
    return df_d
        
def make_tree(node,level,x,y,linestyle,rectstyle,textstyle,max_depth):
    if not isinstance(node, dict):
        return [{**textstyle,'x':[x],'y':[y],'text':["{0:.2f}".format(node)]},
                {**rectstyle,'type':'rect','x0':x-0.4/(2**max_depth),'x1':x+.4/(2**max_depth),'y0':y+0.1/(max_depth),'y1':y-0.1/(max_depth)}]
    
    shapes=[{**rectstyle,'type':'rect','x0':x-0.4/(2**max_depth),'x1':x+.4/(2**max_depth),'y0':y+0.1/(max_depth),'y1':y-0.1/(max_depth)}]
    shapes += [{**textstyle,'x':[x],'y':[y],'text':[('<'+"{0:.2f}".format(node['cut']))]}]
    if 'left' in node:
        shapes+=[{**linestyle,'x0':x,'x1':x-0.5/(2**level),'y0':y,'y1':y-1/(max_depth+1)}]
        shapes+=make_tree(node['left'],level+1,x-0.5/(2**level),y-1/(max_depth+1),linestyle,rectstyle,textstyle,max_depth)
    else:
        nodeL = None
    if 'right' in node:
        shapes+=[{**linestyle,'x0':x,'x1':x+0.5/(2**level),'y0':y,'y1':y-1/(max_depth+1)}]
        shapes+=make_tree(node['right'],level+1,x+0.5/(2**level),y-1/(max_depth+1),linestyle,rectstyle,textstyle,max_depth)
    else:
        nodeR = None
    return shapes


    
colorscale=['rgb(166,206,227)','rgb(31,120,180)','rgb(178,223,138)','rgb(51,160,44)','rgb(251,154,153)','rgb(227,26,28)']  

seed=0
b={}
n_tree=0
model_label='Baseline Model'
N=1000
n=250
df_plot = generate_data('regression','sine',1,250)
df_preds = generate_data('regression','sine',0,N)
## model settings
settings= {'max_depth':1,
           'loss':mse,
           'transform':None,
           'learn_rate':1,
           'min_child_weight':1,
           'l2':0,
           'min_split_loss':0
           }




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

data_style={'width': '250px','display': 'inline-block', 'float': 'left','vertical-align': 'middle','margin-right':20}
slider_style={'height': '20px', 'width': '100%','display': 'inline-block'}
app.layout = html.Div([
    html.H1(children='XGBoost Explorer',style={'text-align':'center','width': '1920px'}),
    html.Div(id='just_created_data',style={'display': 'none'},children='page_load'),
    html.Div(id='just_created_booster',style={'display': 'none'},children='page_load'),
    html.Div(id='just_saved_booster',style={'display': 'none'},children='page_load'),
    html.Div(id='just_updated_model_settings',style={'display':'none'},children='page_load'),
    html.Div(id='just_updated_tree_settings',style={'display':'none'},children='page_load'),
    html.Div(id='just_reset',style={'display':'none'},children='page_load'),
    dcc.Store(id='boosters'),
    html.Div(
        className='row',
        children=[
            html.Div(
                style={'width': '1200px','height':'1100px','display': 'inline-block','float':'left','vertical-align': 'top'},
                children=[
                    html.Div(
                        className='row',
                        style={'width': '1200px','height':'500px','display': 'inline-block','float':'left','vertical-align': 'top'},
                        children=[
                            html.Div(
                                style={'width': '200px','display': 'inline-block', 'float': 'left','vertical-align': 'middle','margin-right':20,'border':'3px solid', 'padding':'10px','margin':'10px'},
                                children=[
                                    html.H3(children='Model Settings'),
                                    html.H6(children='Problem Type'),
                                    dcc.RadioItems(id='model_problem',options=[{'label': 'Classification', 'value': 'classification'},{'label': 'Regression', 'value': 'regression'}],
                                                           value='regression',labelStyle={'display': 'inline-block'}),
                                    html.H6(children='Function'),
                                    html.Div(dcc.Dropdown(id='data_function',options=[{'label':'Sine Wave','value':'sine'},{'label':'Line','value':'line'},{'label':'Sawtooth Wave','value':'saw'},{'label':'Step','value':'step'}],value='sine',clearable=False)),
                                    html.H6(children='Noise'),
                                    html.Div(dcc.Slider(id='data_noise',disabled=False,min=0,max=10,step=0.1,value=2,marks={i: str(i) for i in [0,5,10]}),style=slider_style),
                                    html.H6(children='Sample Size'),
                                    html.Div(dcc.Slider(id='data_size',disabled=False,min=50,max=500,step=50,value=250,marks={i: str(i) for i in [50,250,500]}),style=slider_style),
                                    html.H6(children='Loss Function'),
                                    html.Div(dcc.Dropdown(id='model_loss',options=[{'label':'Mean Squared Error','value':'mse'},{'label':'Logarithmic Loss','value':'logloss'}],value='mse',clearable=False))
                                ]
                            ),
                            html.Div(
                                style={'width': '800px','height':'500px','display': 'inline-block','float':'left','vertical-align': 'top'},
                                children=[dcc.Graph(id='model_plot',animate=True )]
                            )
                        ]
                    ),
                    html.Div(
                        className='row',
                        style={'width': '1200px','height': '500','display': 'inline-block','float':'left','vertical-align': 'top'},
                        children=[
                            html.Div(
                             style={'width': '200px','display': 'inline-block','float':'left','vertical-align': 'top','margin-right':10,'border':'3px solid', 'padding':'10px','margin':'10px'},
                            children=[
                                html.H3('Tree Settings'),
                                 html.H6('Max Depth'),
                                                html.Div(
                                    style=slider_style,
                                    children=[
                                        html.Div(dcc.RadioItems(id='tree_depth',options=[{'label':'1','value':'1'},{'label':'2','value':'2'},{'label':'3','value':'3'}],value='1',labelStyle={'display': 'inline-block'}))
                                    ]
                                ),
                                html.H6('Learning Rate'),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        dcc.Slider(id='tree_learn_rate', marks={float(i/4.0): str(float(i/4.0)) for i in [0,1,2,3,4]},min=0,max=1,value=1,step=0.05)
                                    ]
                                ),
                                html.H6('Min Child Weight'),
                                                html.Div(
                                    style=slider_style,
                                    children=[
                                        dcc.Slider(id='tree_min_child_weight', marks={int(i): str(int(i)) for i in [0,10,25,50]},min=0,max=50,value=1,step=0.5)
                                    ]
                                ),
                                html.H6('L2 Regularization'),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        dcc.Slider(id='tree_l2', marks={float(i): str(int(i)) for i in [0,25,50,100]},min=0,max=100,value=1,step=0.5)
                                    ]
                                ),
                                html.H6('Min Split Loss'),
                                html.Div(
                                    style=slider_style,
                                    children=[
                                        dcc.Slider(id='tree_min_split_loss', marks={float(i): str(int(i)) for i in [0,0.25,0.5,1]},min=0,max=1,value=0,step=0.01)
                                    ]
                                ),
                                html.Div(html.Button('Save Booster',id='save_booster'))
                        ]),
                    html.Div(dcc.Graph(id='booster_plot',animate=True,animation_options={'frame':{'redraw':True}},style={'width': '800px','height':'500px','display': 'inline-block', 'float': 'left','vertical-align': 'top'}))
                            
                        
                    ]
                  
                )]
            ),
            html.Div(dcc.Graph(id='tree_plot',style={'width': '600px','height':'1000px','display': 'inline-block', 'float': 'left','vertical-align': 'top'}))
        ]
    )
])



@app.callback([Output(component_id='just_created_data',component_property='children')],
              [Input(component_id='data_noise', component_property='value'),
               Input(component_id='data_size', component_property='value'),
               Input(component_id='data_function',component_property='value'),
               Input(component_id='model_problem', component_property='value'),
               Input(component_id='model_loss', component_property='value')])
def update_data(noise,size,function,problem,loss):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]
    print('updated data')
    #make animation look nice by reseting noise when changing data points
    if trigger in ['data_noise.value','data_size.value']:
        global seed
        seed+=1
    global df_plot
    global df_preds
    global settings
    
    #save data settings
    if problem == 'regression':
        settings['transform']=None
    elif problem == 'classification':
        settings['transform']=expit
    if loss == 'mse':
        settings['loss']=mse
    elif loss == 'logloss':
        settings['loss']=logloss
    
    #prep dfs
    df_plot=generate_data(problem,function,noise,size)
    df_preds = generate_data(problem,function,0,N)
    df_preds['yhat']=df_plot.loc[0,'yhat']
    df_preds['preds']=transform(df_preds['yhat'])
    return [np.nan]



@app.callback([Output(component_id='just_reset',component_property='children')],
              [Input(component_id='just_created_data', component_property='children'),
               Input(component_id='just_updated_model_settings', component_property='children')])
def reset_all(update1,update2):
    global b
    if len(b)==0:
        raise PreventUpdate
    print('reseting')
    b={}
    return [np.nan]


@app.callback([Output(component_id='just_created_booster',component_property='children')],
              [Input(component_id='just_reset', component_property='children'),
               Input(component_id='just_saved_booster', component_property='children'),
               Input(component_id='just_updated_tree_settings', component_property='children')])
def create_booster(update1,update2,update3):
    global b
    print('creating booster')
    b[n_tree]=Booster(df_plot,settings)
    return [np.nan]
    


    
    
@app.callback([Output(component_id='just_updated_tree_settings',component_property='children')],
              [Input(component_id='tree_depth', component_property='value'),
               Input(component_id='tree_learn_rate', component_property='value'),
               Input(component_id='tree_min_child_weight', component_property='value'),
               Input(component_id='tree_l2', component_property='value'),
               Input(component_id='tree_min_split_loss',component_property='value')])
def update_tree_settings(depth,learn_rate,min_child_weight,l2,min_split_loss):
    if len(b)==0:
        raise PreventUpdate
    print('updating tree settings')
    global settings
    settings['max_depth'] =int(depth) 
    settings['l2'] =float(l2)
    settings['min_child_weight']=float(min_child_weight)
    settings['learn_rate']=float(learn_rate)
    settings['min_split_loss']=float(min_split_loss)
    return [np.nan]
    



        
@app.callback([Output(component_id='just_saved_booster',component_property='children')],
              [Input(component_id='save_booster', component_property='n_clicks')])
def save_booster(click):
    global b
    if len(b)==0 or click is None:
        raise PreventUpdate
    print('saving booster')
    global n_tree
    global settings
    global df_plot
    global df_preds
    df_plot['yhat'] = b[n_tree].yhat
    df_preds['yhat'] = df_preds['yhat']+b[n_tree].predict(df_preds['x'])*settings['learn_rate']
    df_preds['preds']=transform(df_preds['yhat'])
    df_plot['preds']= transform(df_plot['yhat'])
    
    n_tree+=1
    #b[n_tree]=Booster(df_plot,settings)
    return [np.nan]
    
    
@app.callback([Output(component_id='booster_plot',component_property='figure'),
              Output(component_id='model_plot',component_property='figure'),
              Output(component_id='tree_plot',component_property='figure')],
             [Input(component_id='just_created_booster',component_property='children')])
def update_plots(update1):
    global b
    if len(b)==0:
        raise PreventUpdate
    print('updating plots')

    cut_name = 'Cuts' if settings['transform'] is None else "Cuts(Logits)"
    #creating booster plot
 #creating model plot
    model_traces=[{'x':df_plot['x'], 'y':df_plot['y'], 'name':'Data Points','mode':'markers','type':'scatter','line':{'color':colorscale[0]}},
                  {'x':df_plot['x'], 'y':df_plot['true'], 'name':'True Model','mode':'lines','type':'scatter','line':{'color':colorscale[1]}},
                  {'x':df_preds['x'], 'y':df_preds['preds'], 'name':'Current Model','mode':'lines','type':'scatter','line':{'color':colorscale[5]}},
                  {'x':df_preds['x'], 'y': transform(df_preds['yhat']+settings['learn_rate']*b[n_tree].predict(df_preds['x'])),
                         'name': 'New Booster','mode':'lines','hoverinfo':'x','fill':'tonexty',
                         'line':{'dash':'dash','color':colorscale[4]},'showlegend':True}]
    
    model_figure = {'data':model_traces,'layout':{'showlegend':True, 'legend':go.layout.Legend(x=0,y=1),'margin':{'l': 40, 'b': 40, 't': 30, 'r': 10},'yaxis':{'range':[-0.4,1.4],'title':'y'},'xaxis':{'title':'x'}}}
    
    

    if b[n_tree].cant_fit:
        return [{'data' : [go.Scatter(mode='text',text='"Min Split Loss" is too high, no splitting is possible', x=[1], y=[1])]},model_figure,{'data' : [go.Scatter( x=[np.nan], y=[np.nan])]}]
    
    preds=b[n_tree].predict(df_preds['x'])
    loss=b[n_tree].get_loss()
    start=loss.isnull().astype(int).diff()==1
    end=loss.isnull().astype(int).diff()==-1
    start=[1]+start[start['loss']==True].index.tolist()
    end=end[end['loss']==True].index.tolist()+[len(loss)]

    booster_traces=[{'x':list(df_plot['x']),'y':list(df_plot['y']-df_plot['yhat']),
                         'name':'Pseudo-Residuals','mode':'markers','hoverinfo':'x','xaxis':'x','yaxis':'y2','line':{'color':colorscale[0]}},
                        {'x':df_plot['x'],'y':loss['loss'],
                         'name':'Loss','mode':'lines','xaxis':'x','yaxis':'y1','line':{'color':colorscale[2]}},
                        {'x':df_preds['x'],'y':preds,
                         'name':cut_name,'mode':'lines','xaxis':'x','yaxis':'y2','line':{'color':colorscale[4]}}]
    cutstyle = {'type': 'line','yref': 'paper','xref': 'x','y0': 0.0, 'y1': 1,'layer': 'above', 'line': {'color': 'rgb(55, 128, 191)','width': 3}}  
    shadestyle = {'type':'rect','fillcolor': 'rgba(235, 235, 235, 0.5)','yref': 'paper','xref': 'x','layer': 'below','line': {
                    'color': 'rgba(0, 0, 0, 0)',
                    'width': 0,
                }}
    booster_traces.append({'x':[5],'y':[-2],'xref':'paper','yref':'paper','mode':'text','text':'grey shadings block out potential cuts due to regularization settings','showlegend':False})

    booster_shapes=[{**cutstyle,'x0':df_preds.loc[c,'x'],'x1':df_preds.loc[c,'x']} for c in np.nonzero(np.diff(preds))[0]]
   
    for i in range(len(start)):
        booster_shapes.append({**shadestyle,'x0':df_plot.loc[start[i]-1,'x'],'x1':df_plot.loc[end[i]-1,'x'],'y0':0,'y1':1})
    #hack because dash "animate" doesnt work well with a variable number of shapes
    for i in range(50-len(booster_shapes)):
        booster_shapes.append(None)
    
    booster_figure = {'data':booster_traces, 'layout':{'title':'New Booster ','clickmode':'event+select','hovermode':'x','showlegend':True, 'spikedistance':-1,
                                                       'xaxis':{'title':'x','showspikes':True,'showline':True,'showgrid':True,'spikethickness':3,
                                                                'spikesnap':"cursor",'spikedash':'solid','spikemode':"across+toaxis"},
                                                       'yaxis':{'title':'loss','domain':[0,0.2]},
                                                       'yaxis2':{'title':'Pseudo Residuals','domain':[0.3,1],'range':[-0.8,0.8]},
                                                       'legend':go.layout.Legend(x=0.06,y=0.3),'margin':{'l': 40, 'b': 40, 't': 40, 'r': 10},
                                                       'shapes': booster_shapes}}

  

    #creating tree plot
    linestyle={'type':'line','layer':'below'}
    rectstyle =  {'fillcolor': 'rgba(255, 255, 255, 1)','layer': 'below'}
    textstyle = {'mode':'text','textposition':'middle center','textfont':{'family':'sans serif','size':18,'color':'#1f77b4'},'showlegend':False}
    draw=make_tree(b[n_tree].tree,1,0.5,1,linestyle,rectstyle,textstyle,3)
    
    shapes = sorted([i for i in draw if 'type' in i], key=lambda k: k['type'])
    text = [i for i in draw if 'mode' in i]
    tree_figure = {'data':text, 'layout':{'margin':{'l': 40, 'b': 40, 't': 40, 'r': 10},'shapes': shapes,
                                   'xaxis': {'showticklabels': False,'showgrid': False,'zeroline': False, 'autorange': True},
                                   'yaxis': {'showticklabels': False,'showgrid': False,'zeroline': False, 'autorange': True}}}

    
    return [booster_figure,model_figure,tree_figure]
if __name__ == '__main__':
    app.run_server(debug=True)
