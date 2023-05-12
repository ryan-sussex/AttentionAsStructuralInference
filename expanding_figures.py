from dataclasses import dataclass
import dataclasses

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

import json
import numpy as np
from plotly.subplots import make_subplots

from enum import Enum

from run_expanding import RecordResult, RecordExperiment, RecordMultipleResults

purple = "rgba(204,204,245,255)"
green = "rgba(191,230,191,255)"


def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in dataclasses.fields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except:
        return d # Not a dataclass field


@dataclass
class PlotlyTraceConfig:
    name: str
    color: str
    marker_size: int = 20
    marker_color: str = "black"


class TraceType(Enum):
    expanding = PlotlyTraceConfig("expanding", green)
    standard = PlotlyTraceConfig("standard", purple)


def calculate_expanding_complexity(
        iterations: np.array, 
        window: np.array, 
        n_tokens: int, 
        dims: int
    ) -> float:
    iterations = np.array(iterations) + 1
    window = np.array(window)
    truncated = np.maximum(28, np.minimum(window,n_tokens))
    expanding_complexity_series = truncated * (dims + iterations)
    return np.mean(expanding_complexity_series)


def average_loss(loss: np.ndarray, n_points=10) -> np.array:
    return loss.reshape(10, -1).mean(axis=1)


def get_loss_trace(loss: np.ndarray, trace_type: PlotlyTraceConfig):
    return go.Scatter(
        x=list(range(len(loss))),
        y=loss,
        name=trace_type.name,
        marker=dict(
            color=trace_type.color,
            opacity=1,
            size=trace_type.marker_size,
            line=dict(width=2)
        ),
        line=dict(color=trace_type.marker_color),
    )


def get_loss_trace_from_experiment(experiment: RecordExperiment):
    loss = np.array(experiment.loss)
    loss = average_loss(loss)
    trace = get_loss_trace(
        loss, 
        trace_type=TraceType[experiment.attention_type].value
    )
    return trace


def get_loss_traces_from_result(record: RecordResult):
    return [
        get_loss_trace_from_experiment(experiment=record.expanding),
        get_loss_trace_from_experiment(experiment=record.standard)
    ]



def get_complexity_bar_trace(complexity, trace_type: PlotlyTraceConfig, geo_p:float):
    return go.Bar(
        x=[f"p = {geo_p} <br> (Avg. distance to target: {int(1/geo_p)})"], 
        y=[complexity], 
        marker=dict(color=trace_type.color, 
            opacity=1,
            line=dict(width=1, color=trace_type.marker_color)
        ),
    )

def get_complexity_from_experiment(experiment: RecordExperiment):
    if TraceType[experiment.attention_type].value == TraceType.expanding.value:
        complexity = calculate_expanding_complexity(
            iterations=np.array(experiment.iterations),
            window=experiment.window,
            n_tokens=experiment.attention_config.n_tokens,
            dims=experiment.attention_config.n_embd
        )
    else:
        complexity = (
            experiment.attention_config.n_tokens 
            * experiment.attention_config.n_embd
        )
    return complexity

def get_bar_trace_from_experiment(experiment: RecordExperiment):
    return get_complexity_bar_trace(
        get_complexity_from_experiment(experiment),
        trace_type=TraceType[experiment.attention_type].value,
        geo_p=experiment.geo_p
    )

def get_bar_traces_from_result(result: RecordResult):
    return [
        get_bar_trace_from_experiment(result.standard),
        get_bar_trace_from_experiment(result.expanding)
    ]


# def get_complexity_bar()


if __name__ == "__main__":
    
    with open("expanding.json", mode="r") as f:
        training_history = json.load(f)
    
    # print(record)
    figures = []
    records = [dataclass_from_dict(RecordResult, record) for record in training_history]
    for record in records:

        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top', 'title': 'Batch'})

        fig.add_trace(
            get_loss_trace_from_experiment(record.expanding),
            secondary_y=True
        )
        fig.add_trace(
            get_loss_trace_from_experiment(record.standard),
            secondary_y=True
        )
        fig.add_traces(
            get_bar_traces_from_result(result=record)
        )
        fig.data[0].update(xaxis='x2')
        fig.data[1].update(xaxis='x2')
        figures.append(fig)
        # print(
        #     get_complexity_from_experiment(record.standard)
        # )
        # print(
        #     get_complexity_from_experiment(record.expanding)
        # )
        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f"d={record.expanding.attention_config.n_embd}, p={record.expanding.geo_p}",
            # xaxis2_title="Batch",
            yaxis_title="MSE",
            # yaxis2_title="Approx. Number of Operations",
            legend=dict(x=0.6, y=.95),
            # legend_x=0, 
            # legend_y=0,
                font=dict(
                # family="Courier New, monospace",
                size=30,
                # color="RebeccaPurple"
            )
        )
        fig.update_layout(layout)
    
        fig.show()


    selfattention_complexity = (
        record.standard.attention_config.n_tokens 
            * record.standard.attention_config.n_embd
    )
    fig = make_subplots(cols=len(figures), rows=1)
    for i, figure in enumerate(figures):
        fig.layout[f"xaxis{i+5}"] = dict({'anchor': f'y{i+1}', 'overlaying': f'x{i+1}', 'side': 'top', "showticklabels":False})
        # fig.layout[f"xaxis{i+1}"].update(title=f"p={records[i].expanding.geo_p}")
        if i >= 0:
            fig.layout[f"yaxis{i+1}"].update(showticklabels=False, range=[0,  1.3 * selfattention_complexity ])
        if i == 0:
            fig.layout[f"yaxis{i+1}"].update(
                tickvals = [selfattention_complexity],
                ticktext = ["Self Attention"],
                showticklabels=True
            )         
            
        for j, trace_data in enumerate(figure["data"]):
            if i > 0 or j > 1:
                trace_data.update(showlegend=False)
            print(trace_data["type"])
            fig.append_trace(trace_data, row=1, col=i+1)
        fig.data[-4].update(xaxis=f"x{i + 5}")
        fig.data[-3].update(xaxis=f"x{i + 5}")

    
    
    
    
        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f"d={record.expanding.attention_config.n_embd}",
            # xaxis2_title="Batch",
            yaxis_title="Relative Complexity",
            # yaxis2_title="Approx. Number of Operations",
            legend=dict(x=0.1, y=1.1),
            # legend_x=0, 
            # legend_y=0,
            #     font=dict(
            #     # family="Courier New, monospace",
            #     size=30,
            #     # color="RebeccaPurple"
            # )
        )
        fig.update_layout(layout)
    
    fig.show()

    fig.write_image(f"expanding_{record.expanding.attention_config.n_embd}.png")
