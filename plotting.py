import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import plot


def plot_async_rewards(all_c_rewards, plot_fn=plot):
    all_c_rewards = np.array(all_c_rewards)
    worker_count = all_c_rewards.shape[1]
    avg_c_rewards = np.mean(all_c_rewards, axis=1)
    data = [
        dict(
            y=avg_c_rewards,
            name="Average"
        )
    ]
    for i, c_reward_data in enumerate(all_c_rewards.T):
        t = dict(
            y=c_reward_data,
            opacity=0.25,
            name="Agent {}".format(i)
        )
        data.append(t)

    layout = go.Layout(
        title="Average Cumulative Reward - {} agents".format(worker_count),
        xaxis=dict(
            title="Training steps"
        ),
        yaxis=dict(
            title="Cumulative Reward"
        )
    )
    fig = go.Figure(
        data=data,
        layout=layout
    )
    plot_fn(fig, filename="async-rewards.html")


def plot_avg_async_rewards(avg_c_rewards, worker_counts, plot_fn=plot):

    data = []
    for i, c_reward_data in enumerate(avg_c_rewards):
        t = dict(
            y=c_reward_data,
            name="{} workers".format(worker_counts[i])
        )
        data.append(t)

    layout = go.Layout(
        title="Average Cumulative Reward by Worker Count",
        xaxis=dict(
            title="Training steps"
        ),
        yaxis=dict(
            title="Average Cumulative Reward"
        )
    )
    fig = go.Figure(
        data=data,
        layout=layout
    )
    plot_fn(fig, filename="avg-async-rewards.html")


def plot_rewards(
    seed,
    random_rewards,
    learned_rewards,
    filename,
    plot_fn=plot
):
    layout = go.Layout(
        title="Cumulative Reward by Step - Seed {}".format(seed),
        xaxis=dict(
            title="Training Step"
        ),
        yaxis=dict(
            title="Cumulative Reward"
        ),
    )

    data = [
        dict(y=random_rewards, name="Random"),
        dict(y=learned_rewards, name="Learned")
    ]

    fig = go.Figure(data, layout)
    plot_fn(fig, filename=filename)


def plot_Q(agent, filename, plot_fn=plot):
    q_table = agent._Q
    max_q = np.max(q_table, axis=1)
    values = max_q.reshape(agent.env.shape)
    flipped = np.flip(values, axis=0)
    text = np.around(flipped, decimals=6)

    colorscale = [
        [0.0, 'rgb(255,255,255)'],
        [1.0, 'rgb(100,10,255)']
    ]

    rows, cols = values.shape
    fig = ff.create_annotated_heatmap(
        flipped,
        x=["Col {}".format(x) for x in range(cols)],
        y=["Row {}".format(rows - 1 - y) for y in range(rows)],
        colorscale=colorscale,
        annotation_text=text,
    )
    fig.layout.title = "Greedy Policy"

    plot_fn(fig, filename=filename)
