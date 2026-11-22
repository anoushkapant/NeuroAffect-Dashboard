import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).with_name('benchmarks.json')

NAVY = '#0B1F33'
BLUE = '#457B9D'
TEAL = '#2A9D8F'
AMBER = '#E9C46A'
CORAL = '#E76F51'
SLATE = '#6C7A89'
INK = '#1F2A37'

MODEL_COLORS = {
    'EEGNet': TEAL,
    'DeepConvNet': BLUE,
    'Conformer': AMBER,
    'EEGNet-Attention-DANN': CORAL,
    'GRU': SLATE,
    'LSTM': '#8D99AE',
    'Vanilla Transformer': '#9B5DE5',
    'CCNN': '#F4A261',
}


@st.cache_data
def load_benchmarks():
    payload = json.loads(DATA_PATH.read_text())
    frame = pd.DataFrame(payload['models'])
    frame['input_shape'] = frame['input_shape'].apply(
        lambda shape: ' x '.join(str(value) for value in shape))
    return payload, frame


def apply_theme():
    st.set_page_config(page_title='NeuroAffect Dashboard',
                       layout='wide',
                       initial_sidebar_state='expanded')
    st.markdown('''
    <style>
      .stApp { background: #F5F7FA; }
      [data-testid="stSidebar"] { background: #0B1F33; }
      [data-testid="stSidebar"] * { color: #E8EEF5 !important; }
      [data-testid="stSidebar"] .stRadio label { font-weight: 600; }
      .block-container { padding-top: 2rem; }
      .hero {
        background: linear-gradient(120deg, #0B1F33 0%, #173A5E 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        color: #FFFFFF;
        margin-bottom: 1.25rem;
      }
      .hero .tag {
        color: #7FD1C1;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        font-weight: 700;
        text-transform: uppercase;
      }
      .hero h1 { margin: 0.35rem 0 0.35rem 0; font-size: 2rem; }
      .hero p { margin: 0; color: #C9D6E4; font-size: 0.95rem; }
      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0 1.2rem 0;
      }
      .kpi {
        background: #FFFFFF;
        border: 1px solid #E3E8EF;
        border-left: 5px solid #457B9D;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 2px 8px rgba(11,31,51,0.05);
      }
      .kpi .label {
        color: #6C7A89;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
      }
      .kpi .value { font-size: 1.35rem; font-weight: 700; color: #0B1F33; }
      .kpi .detail { color: #6C7A89; font-size: 0.78rem; }
      .insight {
        background: #FFFFFF;
        border: 1px solid #E3E8EF;
        border-left: 5px solid #2A9D8F;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
      }
      .insight strong { color: #0B1F33; }
      .footer {
        color: #8D99AE;
        font-size: 0.75rem;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
      }
    </style>
    ''', unsafe_allow_html=True)


def hero(title, subtitle, tag):
    st.markdown(f'''
    <div class="hero">
      <div class="tag">{tag}</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    ''', unsafe_allow_html=True)


def kpi_row(items):
    cards = []
    for item in items:
        cards.append(f'''
        <div class="kpi" style="border-left-color:{item['accent']}">
          <div class="label">{item['label']}</div>
          <div class="value">{item['value']}</div>
          <div class="detail">{item.get('detail', '')}</div>
        </div>
        ''')
    st.markdown('<div class="kpi-grid">' + ''.join(cards) + '</div>',
                unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="footer">NeuroAffect | Iterum research preview | '
                'Benchmarks from synthetic CPU tensors | Emotion values illustrative'
                '</div>', unsafe_allow_html=True)


def style_figure(figure, height=360):
    figure.update_layout(
        template='plotly_white',
        height=height,
        font=dict(family='Arial, sans-serif', size=12, color=INK),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=30, t=55, b=45),
        hoverlabel=dict(bgcolor=NAVY, font_color='white'),
        legend=dict(orientation='h', y=1.16, x=0),
    )
    figure.update_xaxes(gridcolor='#EDF1F5', zeroline=False)
    figure.update_yaxes(gridcolor='#EDF1F5', zeroline=False)
    return figure


def gauge_figure(value, title, accent):
    figure = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        number={'suffix': '', 'font': {'size': 30, 'color': NAVY}},
        title={'text': title, 'font': {'size': 13, 'color': SLATE}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': SLATE},
            'bar': {'color': accent, 'thickness': 0.28},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [-1, -0.33], 'color': '#E8EEF5'},
                {'range': [-0.33, 0.33], 'color': '#D6E4F0'},
                {'range': [0.33, 1], 'color': '#CBE5DF'},
            ],
        }))
    return style_figure(figure, 260)


def circumplex_figure(valence, arousal):
    figure = go.Figure()
    quadrants = [
        (-1, 0, 0, 1, '#FDF2F0', 'Negative / High arousal'),
        (0, 0, 1, 1, '#FFF9E8', 'Positive / High arousal'),
        (-1, -1, 0, 0, '#F1F4F8', 'Negative / Low arousal'),
        (0, -1, 1, 0, '#EDF7F4', 'Positive / Low arousal'),
    ]
    for x0, y0, x1, y1, color, label in quadrants:
        figure.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1,
                         fillcolor=color, line=dict(width=0), layer='below')
        figure.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                              text=label, showarrow=False,
                              font=dict(size=10, color=SLATE))
    figure.add_shape(type='line', x0=0, y0=-1, x1=0, y1=1,
                     line=dict(color='#B8C4D0', width=1))
    figure.add_shape(type='line', x0=-1, y0=0, x1=1, y1=0,
                     line=dict(color='#B8C4D0', width=1))
    figure.add_trace(go.Scatter(
        x=[valence], y=[arousal],
        mode='markers+text',
        text=['Current state'],
        textposition='top center',
        marker=dict(size=20, color=CORAL, line=dict(color='white', width=2)),
        showlegend=False,
    ))
    figure.update_xaxes(title='Valence', range=[-1, 1], dtick=0.5)
    figure.update_yaxes(title='Arousal', range=[-1, 1], dtick=0.5)
    return style_figure(figure, 300)


def metric_bar_figure(frame, column, title, accent, fmt):
    ordered = frame.sort_values(column, ascending=True)
    labels = [fmt(value) for value in ordered[column]]
    figure = go.Figure(go.Bar(
        y=ordered['model'],
        x=ordered[column],
        orientation='h',
        text=labels,
        textposition='outside',
        cliponaxis=False,
        marker=dict(color=[MODEL_COLORS.get(model, accent)
                           for model in ordered['model']]),
    ))
    figure.update_xaxes(title=title)
    figure.update_yaxes(title='')
    return style_figure(figure, 360)


def efficiency_scores(frame):
    scores = pd.DataFrame(index=frame.index)
    for column in ['params', 'gflops', 'ms_per_batch']:
        values = np.log10(frame[column].astype(float))
        span = values.max() - values.min()
        scores[column] = 1 - (values - values.min()) / span
    return scores


def profile_figure(frame, selected_model):
    scores = efficiency_scores(frame)
    axes = ['Size efficiency', 'Compute efficiency', 'Latency efficiency']
    selected = frame[frame['model'] == selected_model].iloc[0]
    baseline = frame[frame['model'] == 'EEGNet'].iloc[0]

    def values_for(row):
        row_scores = scores.loc[row.name]
        return [row_scores['params'], row_scores['gflops'],
                row_scores['ms_per_batch']]

    figure = go.Figure()
    figure.add_trace(go.Scatterpolar(
        r=values_for(selected),
        theta=axes,
        fill='toself',
        name=selected_model,
        line=dict(color=MODEL_COLORS.get(selected_model, CORAL), width=2),
    ))
    figure.add_trace(go.Scatterpolar(
        r=values_for(baseline),
        theta=axes,
        fill='toself',
        name='EEGNet baseline',
        line=dict(color=BLUE, width=2, dash='dot'),
    ))
    figure.add_trace(go.Scatterpolar(
        r=[scores[col].max() for col in ['params', 'gflops', 'ms_per_batch']],
        theta=axes,
        name='Best measured',
        line=dict(color=AMBER, width=2, dash='dash'),
    ))
    figure.update_polars(radialaxis=dict(range=[0, 1], visible=True,
                                        tickvals=[0, 0.5, 1]))
    figure.update_layout(polar=dict(bgcolor='white'), height=360)
    figure.update_layout(
        font=dict(family='Arial, sans-serif', size=12, color=INK),
        paper_bgcolor='white',
        legend=dict(orientation='h', y=1.12, x=0),
    )
    return figure


def frontier_points(frame):
    ordered = frame.sort_values('params')
    best = float('inf')
    points = []
    for _, row in ordered.iterrows():
        if row['ms_per_batch'] <= best:
            points.append(row)
            best = row['ms_per_batch']
    return pd.DataFrame(points)


def frontier_figure(frame):
    frontier = frontier_points(frame)
    figure = go.Figure()
    for _, row in frame.iterrows():
        color = MODEL_COLORS.get(row['model'], SLATE)
        figure.add_trace(go.Scatter(
            x=[row['params']],
            y=[row['ms_per_batch']],
            mode='markers+text',
            text=[row['model']],
            textposition='top center',
            marker=dict(size=11 + 6 * np.sqrt(row['gflops']),
                        color=color,
                        line=dict(color='white', width=1.5)),
            name=row['model'],
            showlegend=False,
            customdata=[[row['gflops']]],
            hovertemplate=(f'{row["model"]}<br>Parameters: '
                           '%{x:,.0f}<br>Latency: %{y:.3f} ms/batch'
                           '<br>GFLOPs: %{customdata[0]:.4f}<extra></extra>'),
        ))
    figure.add_trace(go.Scatter(
        x=frontier['params'],
        y=frontier['ms_per_batch'],
        mode='lines+markers',
        line=dict(color=NAVY, width=2, dash='dash'),
        marker=dict(size=9, color=NAVY),
        name='Efficiency frontier',
    ))
    figure.update_xaxes(type='log', title='Parameters (log scale)')
    figure.update_yaxes(title='Latency (ms/batch)')
    return style_figure(figure, 430)


def waterfall_figure(frame):
    baseline = frame[frame['model'] == 'EEGNet']['params'].iloc[0]
    proposed = frame[frame['model'] == 'EEGNet-Attention-DANN']['params'].iloc[0]
    figure = go.Figure(go.Waterfall(
        measure=['absolute', 'relative', 'total'],
        x=['EEGNet', 'Attention + domain head', 'EEGNet-Attention-DANN'],
        y=[baseline, proposed - baseline, proposed],
        text=[f'{baseline:,}', f'+{proposed - baseline:,}', f'{proposed:,}'],
        textposition='outside',
        connector=dict(line=dict(color='#B8C4D0', width=1)),
        increasing=dict(marker=dict(color=CORAL)),
        totals=dict(marker=dict(color=NAVY)),
        showlegend=False,
    ))
    figure.update_yaxes(title='Parameters', type='log')
    return style_figure(figure, 330)


def architecture_flow_figure():
    stages = [
        ('Input', '1 x 32 x 128', '#E8EEF5'),
        ('Temporal conv', 'frequency features', '#D6E4F0'),
        ('Spatial conv', 'channel mixing', '#D6E4F0'),
        ('Attention', 'temporal tokens', '#CBE5DF'),
        ('Domain head', 'gradient reversal', '#F8DFD8'),
        ('Classifier', '2 classes', '#E8EEF5'),
    ]
    figure = go.Figure()
    for index, (name, detail, color) in enumerate(stages):
        x0 = index * 1.9
        figure.add_shape(type='rect', x0=x0, x1=x0 + 1.45, y0=0, y1=1,
                         fillcolor=color,
                         line=dict(color=BLUE if index < 4 else CORAL, width=2))
        figure.add_annotation(x=x0 + 0.725, y=0.63, text=name,
                              showarrow=False, font=dict(size=12, color=NAVY))
        figure.add_annotation(x=x0 + 0.725, y=0.3, text=detail,
                              showarrow=False, font=dict(size=9, color=SLATE))
        if index < len(stages) - 1:
            figure.add_annotation(x=x0 + 1.68, y=0.5, text='>',
                                  showarrow=False,
                                  font=dict(size=18, color=NAVY))
    figure.update_xaxes(range=[-0.2, 10.5], visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                         paper_bgcolor='white', plot_bgcolor='white')
    return figure


def training_flow_figure():
    figure = go.Figure()
    nodes = [
        (0.1, 0.72, 'Source batch', 'labelled emotion', '#D6E4F0'),
        (0.1, 0.18, 'Target batch', 'unlabelled subject', '#F8DFD8'),
        (2.3, 0.45, 'Shared encoder', 'EEGNet + attention', '#CBE5DF'),
        (4.6, 0.72, 'Classifier', 'emotion loss', '#D6E4F0'),
        (4.6, 0.18, 'Domain head', 'adversarial loss', '#F8DFD8'),
    ]
    for x, y, title, detail, color in nodes:
        figure.add_shape(type='rect', x0=x, x1=x + 1.6, y0=y, y1=y + 0.22,
                         fillcolor=color,
                         line=dict(color=NAVY, width=1.5))
        figure.add_annotation(x=x + 0.8, y=y + 0.15, text=title,
                              showarrow=False, font=dict(size=11, color=NAVY))
        figure.add_annotation(x=x + 0.8, y=y + 0.05, text=detail,
                              showarrow=False, font=dict(size=9, color=SLATE))
    arrows = [
        (1.7, 0.83, 2.3, 0.58),
        (1.7, 0.29, 2.3, 0.52),
        (3.9, 0.58, 4.6, 0.83),
        (3.9, 0.52, 4.6, 0.29),
    ]
    for x0, y0, x1, y1 in arrows:
        figure.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                              showarrow=True, arrowhead=2,
                              arrowcolor=NAVY, arrowsize=1.2,
                              arrowwidth=1.5)
    figure.add_annotation(x=4.6, y=0.02,
                          text='Gradient reversal discourages subject-specific features',
                          showarrow=False, font=dict(size=10, color=CORAL))
    figure.update_xaxes(range=[0, 6.5], visible=False)
    figure.update_yaxes(range=[-0.05, 1.1], visible=False)
    figure.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                         paper_bgcolor='white', plot_bgcolor='white')
    return figure


def overview_page(payload, frame, selected_model):
    valence, arousal = 0.65, 0.72
    selected = frame[frame['model'] == selected_model].iloc[0]
    hero('NeuroAffect Dashboard',
         'Subject-invariant EEG emotion recognition with a compact '
         'attention-domain adaptation pipeline.',
         'Iterum research preview')

    kpi_row([
        {'label': 'Current emotion', 'value': 'Positive / High arousal',
         'detail': 'Illustrative inference state', 'accent': CORAL},
        {'label': 'Confidence', 'value': '89.2%',
         'detail': 'Illustrative model output', 'accent': TEAL},
        {'label': 'Subject', 'value': 'S12',
         'detail': 'Held-out subject example', 'accent': BLUE},
        {'label': 'Selected model', 'value': selected_model,
         'detail': 'Choose another in the sidebar', 'accent': AMBER},
    ])

    left, right = st.columns(2)
    with left:
        st.plotly_chart(gauge_figure(valence, 'Valence', BLUE),
                        width='stretch')
        st.plotly_chart(gauge_figure(arousal, 'Arousal', TEAL),
                        width='stretch')
    with right:
        st.plotly_chart(circumplex_figure(valence, arousal),
                        width='stretch')

    st.subheader('Selected model snapshot')
    kpi_row([
        {'label': 'Parameters', 'value': f'{selected["params"]:,}',
         'detail': 'Forward-pass weights', 'accent': BLUE},
        {'label': 'Compute', 'value': f'{selected["gflops"]:.4f}',
         'detail': 'GFLOPs per sample batch', 'accent': TEAL},
        {'label': 'Latency', 'value': f'{selected["ms_per_batch"]:.3f} ms',
         'detail': f'CPU, batch {payload["batch_size"]}', 'accent': AMBER},
        {'label': 'Input shape', 'value': selected['input_shape'],
         'detail': 'Batch x channels x time', 'accent': SLATE},
    ])

    insight('Emotion, confidence and subject values are illustrative. '
            'Model cost figures are measured from synthetic forward passes.')


def comparison_page(frame):
    fastest = frame.loc[frame['ms_per_batch'].idxmin()]
    smallest = frame.loc[frame['params'].idxmin()]
    baseline = frame[frame['model'] == 'EEGNet']['params'].iloc[0]
    proposed = frame[frame['model'] == 'EEGNet-Attention-DANN']['params'].iloc[0]

    hero('Model Comparison',
         'Decision view of cost, latency and efficiency across the '
         'benchmarked families.',
         'Benchmark intelligence')

    kpi_row([
        {'label': 'Models benchmarked', 'value': str(len(frame)),
         'detail': 'CNN, RNN and Transformer families', 'accent': BLUE},
        {'label': 'Smallest', 'value': smallest['model'],
         'detail': f'{smallest["params"]:,} parameters', 'accent': TEAL},
        {'label': 'Fastest', 'value': fastest['model'],
         'detail': f'{fastest["ms_per_batch"]:.3f} ms/batch', 'accent': AMBER},
        {'label': 'Proposed overhead', 'value': f'+{proposed - baseline:,}',
         'detail': 'Parameters over EEGNet', 'accent': CORAL},
    ])

    trade_tab, profile_tab, table_tab = st.tabs([
        'Efficiency frontier', 'Efficiency profile', 'Ranked table'])

    with trade_tab:
        st.plotly_chart(frontier_figure(frame), width='stretch')
        frontier = frontier_points(frame)
        names = ', '.join(frontier['model'].tolist())
        insight(f'The measured efficiency frontier contains: <strong>{names}</strong>. '
                'Points below and to the left offer a better cost-latency trade-off.')

    with profile_tab:
        selected_model = st.selectbox('Profile model', frame['model'].tolist(),
                                      index=3, key='profile_model')
        st.plotly_chart(profile_figure(frame, selected_model),
                        width='stretch')
        st.caption('Scores are derived from measured benchmarks. Higher is '
                   'better; each axis is normalised across all models.')

    with table_tab:
        scores = efficiency_scores(frame)
        ranked = frame.copy()
        ranked['Efficiency index'] = scores.mean(axis=1).round(3)
        display = ranked[[
            'model', 'params', 'gflops', 'ms_per_batch', 'Efficiency index']]
        display = display.sort_values('Efficiency index', ascending=False)
        display = display.rename(columns={
            'model': 'Model',
            'params': 'Parameters',
            'gflops': 'GFLOPs',
            'ms_per_batch': 'Latency (ms/batch)',
        })
        st.dataframe(display.style.format({
            'Parameters': '{:,}',
            'GFLOPs': '{:.4f}',
            'Latency (ms/batch)': '{:.3f}',
            'Efficiency index': '{:.3f}',
        }).bar('Efficiency index', color='#CBE5DF'),
            hide_index=True, width='stretch')


def architecture_page(frame):
    proposed = frame[frame['model'] == 'EEGNet-Attention-DANN'].iloc[0]
    baseline = frame[frame['model'] == 'EEGNet'].iloc[0]

    hero('Architecture',
         'EEGNet backbone with temporal attention and an adversarial '
         'domain head for subject invariance.',
         'Model design')

    forward_tab, training_tab, ablation_tab = st.tabs([
        'Forward path', 'Training path', 'Ablation'])

    with forward_tab:
        st.plotly_chart(architecture_flow_figure(), width='stretch')
        st.markdown('''
        - **EEGNet backbone** extracts compact temporal and spatial features.
        - **Multi-head attention** weights the temporal feature sequence.
        - **Domain head** receives features through a gradient reversal layer.
        - **Classifier** predicts the two emotion classes.
        ''')

    with training_tab:
        st.plotly_chart(training_flow_figure(), width='stretch')
        st.caption('Source subjects provide labelled emotion supervision; '
                   'the held-out subject contributes unlabelled target data '
                   'to the domain adversary.')

    with ablation_tab:
        left, right = st.columns([1.1, 1])
        with left:
            st.plotly_chart(waterfall_figure(frame), width='stretch')
        with right:
            ablation = pd.DataFrame([
                {'Variant': 'EEGNet', 'Attention': 'No', 'DANN': 'No',
                 'Parameter delta': '0'},
                {'Variant': 'EEGNet + Attention', 'Attention': 'Yes',
                 'DANN': 'No', 'Parameter delta': 'Not benchmarked'},
                {'Variant': 'EEGNet + DANN', 'Attention': 'No',
                 'DANN': 'Yes', 'Parameter delta': 'Not benchmarked'},
                {'Variant': 'EEGNet-Attention-DANN', 'Attention': 'Yes',
                 'DANN': 'Yes',
                 'Parameter delta': f'+{proposed["params"] - baseline["params"]:,}'},
            ])
            st.dataframe(ablation, hide_index=True, width='stretch')
            st.caption('Only the baseline and proposed variants were '
                       'included in the synthetic benchmark run.')


def benchmarks_page(payload, frame):
    hero('Synthetic Benchmarks',
         f'Measured on {payload["device"]} with batch size '
         f'{payload["batch_size"]}. Forward pass only, random tensors.',
         'Reproducible measurements')

    selected = st.multiselect('Models', frame['model'].tolist(),
                              default=frame['model'].tolist(),
                              key='benchmark_models')
    filtered = frame[frame['model'].isin(selected)]
    if filtered.empty:
        st.warning('Select at least one model.')
        return

    left, right = st.columns(2)
    with left:
        st.plotly_chart(metric_bar_figure(
            filtered, 'params', 'Parameters', BLUE,
            lambda value: f'{value:,.0f}'), width='stretch')
    with right:
        st.plotly_chart(metric_bar_figure(
            filtered, 'gflops', 'GFLOPs', TEAL,
            lambda value: f'{value:.4f}'), width='stretch')

    st.plotly_chart(metric_bar_figure(
        filtered, 'ms_per_batch', 'Latency (ms/batch)', AMBER,
        lambda value: f'{value:.3f}'), width='stretch')

    display = filtered[[
        'model', 'params', 'gflops', 'ms_per_batch', 'input_shape']].rename(
            columns={
                'model': 'Model',
                'params': 'Parameters',
                'gflops': 'GFLOPs',
                'ms_per_batch': 'Latency (ms/batch)',
                'input_shape': 'Input shape',
            })
    st.dataframe(display.style.format({
        'Parameters': '{:,}',
        'GFLOPs': '{:.4f}',
        'Latency (ms/batch)': '{:.3f}',
    }), hide_index=True, width='stretch')

    st.download_button('Download benchmarks as CSV',
                       filtered.to_csv(index=False).encode('utf-8'),
                       file_name='neuroaffect_benchmarks.csv',
                       mime='text/csv')


def main():
    apply_theme()
    payload, frame = load_benchmarks()

    page = st.sidebar.radio('Navigation', [
        'Overview',
        'Model Comparison',
        'Architecture',
        'Synthetic Benchmarks',
    ])
    selected_model = st.sidebar.selectbox('Focus model',
                                          frame['model'].tolist(),
                                          index=3)
    st.sidebar.divider()
    st.sidebar.caption('Iterum edition. Live inference and DEAP results are '
                       'not connected.')

    if page == 'Overview':
        overview_page(payload, frame, selected_model)
    elif page == 'Model Comparison':
        comparison_page(frame)
    elif page == 'Architecture':
        architecture_page(frame)
    else:
        benchmarks_page(payload, frame)

    footer()


if __name__ == '__main__':
    main()
