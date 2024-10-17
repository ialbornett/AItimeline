import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# Data: List of AI Milestones with Descriptions
data = [
    {
        "Year": 1950,
        "Milestone": "Turing Test Proposed",
        "Main_Persons": "Alan Turing",
        "Description": "Alan Turing publishes 'Computing Machinery and Intelligence', introducing the concept of the Turing Test to determine if a machine can exhibit human-like intelligence."
    },
    {
        "Year": 1952,
        "Milestone": "First Self-Learning Program (Checkers)",
        "Main_Persons": "Arthur Samuel",
        "Description": "Arthur Samuel develops a checkers-playing program that learns and improves over time, demonstrating an early form of machine learning."
    },
    {
        "Year": 1956,
        "Milestone": "Dartmouth Conference (Birth of AI)",
        "Main_Persons": "John McCarthy, Marvin Minsky, Claude Shannon, Nathaniel Rochester",
        "Description": "The Dartmouth Conference is held, where the term 'Artificial Intelligence' is coined, marking the founding of AI as a field."
    },
    # ... (Include all other milestones with descriptions as provided previously)
    {
        "Year": 2023,
        "Milestone": "GPT-4 Released",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases GPT-4, an advanced language model with enhanced capabilities in understanding and generating text."
    },
]

# Create a DataFrame
df = pd.DataFrame(data)

# Sort the DataFrame by Year
df = df.sort_values(by="Year").reset_index(drop=True)

# Calculate cumulative milestones
df["Cumulative"] = range(1, len(df) + 1)

# Streamlit App
st.set_page_config(page_title="AI Milestones Timeline", layout="wide")

st.title("Significant AI Milestones Timeline")

# Filter milestones based on year range
st.sidebar.header("Filter Milestones")
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

filtered_df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])].reset_index(drop=True)

# Create cumulative milestones line chart
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Year"],
        y=filtered_df["Cumulative"],
        mode="lines+markers",
        marker=dict(size=8, color="DarkBlue"),
        line=dict(color="DarkBlue"),
        hovertext=filtered_df.apply(
            lambda row: f"<b>Year:</b> {row['Year']}<br>"
                        f"<b>Cumulative Milestones:</b> {row['Cumulative']}<br>"
                        f"<b>Milestone:</b> {row['Milestone']}",
            axis=1
        ),
        hoverinfo="text",
    )
)

# Update layout
fig.update_layout(
    height=500,
    showlegend=False,
    xaxis_title="Year",
    yaxis_title="Cumulative Milestones",
    xaxis=dict(range=[year_range[0] - 1, year_range[1] + 1]),
    yaxis=dict(range=[0, filtered_df["Cumulative"].max() + 1]),
    hovermode="closest",
)

# Display the cumulative milestones line chart and capture click events
selected_points = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=500,
    override_width="100%",
)

# Show detailed information when a milestone is selected
st.header("Milestone Details")

if selected_points:
    # Get the index of the selected point
    point_index = selected_points[0]["pointIndex"]
    milestone_details = filtered_df.iloc[point_index]
    
    st.markdown(f"**Year:** {milestone_details['Year']}")
    st.markdown(f"**Milestone:** {milestone_details['Milestone']}")
    st.markdown(f"**Main Person(s):** {milestone_details['Main_Persons']}")
    st.markdown(f"**Description:** {milestone_details['Description']}")
else:
    st.write("Click on a point in the chart to see the milestone details.")
