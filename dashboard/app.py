import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).with_name('benchmarks.json')


@st.cache_data
def load_benchmarks():
    payload = json.loads(DATA_PATH.read_text())
    frame = pd.DataFrame(payload['models'])
    frame['input_shape'] = frame['input_shape'].apply(
        lambda shape: ' x '.join(str(value) for value in shape))
    return payload, frame


def indicator(value, title, low=None, high=None, suffix=''):
    figure = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        number={'suffix': suffix},
        title={'text': title},
        gauge={
            'axis': {'range': [low, high]},
            'bar': {'color': '#4C72B0'},
            'steps': [
                {'range': [low, low + (high - low) / 3], 'color': '#E8F0FE'},
                {'range': [low + (high - low) / 3, high - (high - low) / 3],
                 'color': '#D4E4FC'},
                {'range': [high - (high - low) / 3, high], 'color': '#B9D5FA'},
            ],
        }))
    figure.update_layout(height=230, margin=dict(t=40, b=10, l=20, r=20))
    return figure


def overview_page(payload, frame, selected_model):
    st.title('NeuroAffect Dashboard')
    st.caption('Iterum demo. Emotion and confidence values are illustrative; '
               'no live inference is connected.')

    metric_col, subject_col, model_col = st.columns(3)
    metric_col.metric('Current emotion', 'Positive / High Arousal')
    subject_col.metric('Subject ID', 'S12')
    model_col.metric('Model', selected_model)

    confidence_col, risk_col = st.columns(2)
    confidence_col.metric('Confidence', '89.2%')
    risk_col.metric('Risk level', 'Normal')

    valence_col, arousal_col = st.columns(2)
    with valence_col:
        st.plotly_chart(indicator(0.65,
                                  'Valence (Negative to Positive)',
                                  low=-1,
                                  high=1),
                        width='stretch')
    with arousal_col:
        st.plotly_chart(indicator(0.72,
                                  'Arousal (Calm to Excited)',
                                  low=-1,
                                  high=1),
                        width='stretch')

    selected = frame[frame['model'] == selected_model].iloc[0]
    st.subheader('Selected model at a glance')
    param_col, flop_col, latency_col = st.columns(3)
    param_col.metric('Parameters', f'{selected["params"]:,}')
    flop_col.metric('GFLOPs', f'{selected["gflops"]:.3f}')
    latency_col.metric('Latency', f'{selected["ms_per_batch"]:.2f} ms/batch')


def comparison_page(frame):
    st.title('Model Comparison')
    st.caption('Synthetic CPU benchmark, batch size 16. No dataset accuracy '
               'is available until DEAP evaluation is run.')

    display = frame[['model', 'params', 'gflops', 'ms_per_batch',
                     'input_shape']].rename(columns={
                         'model': 'Model',
                         'params': 'Parameters',
                         'gflops': 'GFLOPs',
                         'ms_per_batch': 'ms/batch',
                         'input_shape': 'Input',
                     })
    st.dataframe(display, hide_index=True, width='stretch')

    sorted_frame = frame.sort_values('params', ascending=True)
    param_fig = go.Figure(go.Bar(x=sorted_frame['model'],
                                 y=sorted_frame['params'],
                                 marker_color='#4C72B0'))
    param_fig.update_yaxes(type='log', title='Parameters (log scale)')
    param_fig.update_layout(title='Parameter count', height=380)
    st.plotly_chart(param_fig, width='stretch')

    pareto_fig = go.Figure(go.Scatter(
        x=frame['params'],
        y=frame['ms_per_batch'],
        mode='markers+text',
        text=frame['model'],
        textposition='top center',
        marker=dict(size=14, color='#55A868'),
    ))
    pareto_fig.update_xaxes(type='log', title='Parameters (log scale)')
    pareto_fig.update_yaxes(title='ms/batch')
    pareto_fig.update_layout(title='Latency vs capacity (synthetic)',
                             height=420)
    st.plotly_chart(pareto_fig, width='stretch')


def architecture_page(frame):
    st.title('Architecture')
    st.caption('EEGNet backbone with temporal multi-head attention and an '
               'adversarial domain head.')

    stages = [
        ('EEG', '1 x 32 x 128'),
        ('EEGNet backbone', 'temporal + spatial conv'),
        ('Multi-head attention', 'temporal tokens'),
        ('DANN head', 'gradient reversal'),
        ('Classification', '2 classes'),
    ]
    figure = go.Figure()
    for index, (name, detail) in enumerate(stages):
        x0 = index * 2.0
        figure.add_shape(type='rect',
                         x0=x0,
                         x1=x0 + 1.5,
                         y0=0,
                         y1=1,
                         fillcolor='#D4E4FC' if index in (2, 3) else '#EDF2F7',
                         line=dict(color='#4C72B0', width=2))
        figure.add_annotation(x=x0 + 0.75,
                              y=0.62,
                              text=name,
                              showarrow=False,
                              font=dict(size=13))
        figure.add_annotation(x=x0 + 0.75,
                              y=0.28,
                              text=detail,
                              showarrow=False,
                              font=dict(size=10))
        if index > 0:
            figure.add_annotation(x=x0 - 0.25,
                                  y=0.5,
                                  text='>',
                                  showarrow=False,
                                  font=dict(size=22))
    figure.update_xaxes(range=[-0.3, 9.8], visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figure, width='stretch')

    st.subheader('Ablation matrix')
    st.dataframe(pd.DataFrame([
        {'Model': 'EEGNet', 'Attention': 'No', 'DANN': 'No'},
        {'Model': 'EEGNet + Attention', 'Attention': 'Yes', 'DANN': 'No'},
        {'Model': 'EEGNet + DANN', 'Attention': 'No', 'DANN': 'Yes'},
        {'Model': 'EEGNet-Attention-DANN', 'Attention': 'Yes', 'DANN': 'Yes'},
    ]),
        hide_index=True,
        width='stretch')

    proposed = frame[frame['model'] == 'EEGNet-Attention-DANN'].iloc[0]
    baseline = frame[frame['model'] == 'EEGNet'].iloc[0]
    st.info(f'The proposed model adds '
            f'{proposed["params"] - baseline["params"]:,} parameters over '
            f'EEGNet for attention and the domain head.')


def benchmarks_page(payload, frame):
    st.title('Synthetic Benchmark Results')
    st.caption(f'Device: {payload["device"]}, batch size: '
               f'{payload["batch_size"]}. Random tensors, forward pass only.')

    selected = st.multiselect('Models', frame['model'].tolist(),
                              default=frame['model'].tolist())
    filtered = frame[frame['model'].isin(selected)]
    if filtered.empty:
        st.warning('Select at least one model.')
        return

    for column, title, log_scale in [
        ('params', 'Parameters', True),
        ('gflops', 'GFLOPs', False),
        ('ms_per_batch', 'Latency (ms/batch)', False),
    ]:
        ordered = filtered.sort_values(column, ascending=True)
        figure = go.Figure(go.Bar(x=ordered['model'],
                                  y=ordered[column],
                                  marker_color='#4C72B0'))
        if log_scale:
            figure.update_yaxes(type='log')
        figure.update_layout(title=title, height=330)
        st.plotly_chart(figure, width='stretch')

    st.dataframe(filtered[['model', 'params', 'gflops', 'ms_per_batch',
                           'input_shape']],
                 hide_index=True,
                 width='stretch')


def main():
    st.set_page_config(page_title='NeuroAffect Dashboard', layout='wide')
    payload, frame = load_benchmarks()
    page = st.sidebar.radio('Page', [
        'Overview',
        'Model Comparison',
        'Architecture',
        'Synthetic Benchmarks',
    ])
    selected_model = st.sidebar.selectbox('Model', frame['model'].tolist(),
                                          index=3)
    st.sidebar.caption('Iterum edition. Live inference and dataset results '
                       'are not connected yet.')

    if page == 'Overview':
        overview_page(payload, frame, selected_model)
    elif page == 'Model Comparison':
        comparison_page(frame)
    elif page == 'Architecture':
        architecture_page(frame)
    else:
        benchmarks_page(payload, frame)


if __name__ == '__main__':
    main()
