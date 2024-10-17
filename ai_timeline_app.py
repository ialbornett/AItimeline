import streamlit as st
import pandas as pd
import plotly.express as px

# Data: List of AI Milestones
data = [
    {"Year": 1950, "Milestone": "Turing Test Proposed", "Main_Persons": "Alan Turing"},
    {"Year": 1952, "Milestone": "First Self-Learning Program (Checkers)", "Main_Persons": "Arthur Samuel"},
    {"Year": 1956, "Milestone": "Dartmouth Conference (Birth of AI)", "Main_Persons": "John McCarthy, Marvin Minsky, Claude Shannon, Nathaniel Rochester"},
    {"Year": 1956, "Milestone": "Logic Theorist (First AI Program)", "Main_Persons": "Allen Newell and Herbert A. Simon"},
    {"Year": 1957, "Milestone": "General Problem Solver", "Main_Persons": "Herbert A. Simon, J.C. Shaw, Allen Newell"},
    {"Year": 1957, "Milestone": "Perceptron Invented", "Main_Persons": "Frank Rosenblatt"},
    {"Year": 1958, "Milestone": "LISP Programming Language", "Main_Persons": "John McCarthy"},
    {"Year": 1965, "Milestone": "ELIZA (First Chatbot)", "Main_Persons": "Joseph Weizenbaum"},
    {"Year": 1969, "Milestone": "Stanford AI Lab Founded", "Main_Persons": "John McCarthy"},
    {"Year": 1970, "Milestone": "First Robot Arm (Shakey)", "Main_Persons": "Charles Rosen, Nils Nilsson"},
    {"Year": 1970, "Milestone": "SHRDLU (Natural Language Understanding)", "Main_Persons": "Terry Winograd"},
    {"Year": 1972, "Milestone": "MYCIN (Expert System for Medical Diagnosis)", "Main_Persons": "Edward Shortliffe"},
    {"Year": 1974, "Milestone": "Backpropagation Algorithm in Neural Networks (Initial Development)", "Main_Persons": "Paul Werbos"},
    {"Year": 1980, "Milestone": "Expert Systems", "Main_Persons": "Edward Feigenbaum"},
    {"Year": 1982, "Milestone": "Hopfield Network", "Main_Persons": "John Hopfield"},
    {"Year": 1986, "Milestone": "Backpropagation for Neural Networks (Popularized)", "Main_Persons": "David Rumelhart, Geoffrey Hinton, Ronald Williams"},
    {"Year": 1995, "Milestone": "Support Vector Machines Popularized", "Main_Persons": "Vladimir Vapnik"},
    {"Year": 1997, "Milestone": "IBM's Deep Blue Defeats Garry Kasparov", "Main_Persons": "IBM Team led by Feng-hsiung Hsu"},
    {"Year": 1997, "Milestone": "Long Short-Term Memory (LSTM) Networks Introduced", "Main_Persons": "Sepp Hochreiter and Jürgen Schmidhuber"},
    {"Year": 2002, "Milestone": "Roomba (First Mass-Market Autonomous Robot)", "Main_Persons": "iRobot Corporation"},
    {"Year": 2006, "Milestone": "Netflix Prize Competition Begins", "Main_Persons": "Netflix"},
    {"Year": 2011, "Milestone": "IBM Watson Wins Jeopardy!", "Main_Persons": "IBM Team"},
    {"Year": 2012, "Milestone": "ImageNet Breakthrough Using Deep Learning (AlexNet)", "Main_Persons": "Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton"},
    {"Year": 2014, "Milestone": "Generative Adversarial Networks (GANs)", "Main_Persons": "Ian Goodfellow"},
    {"Year": 2016, "Milestone": "AlphaGo Defeats Lee Sedol", "Main_Persons": "DeepMind Team led by Demis Hassabis"},
    {"Year": 2017, "Milestone": "Transformer Model Introduced", "Main_Persons": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al."},
    {"Year": 2018, "Milestone": "BERT Language Model", "Main_Persons": "Google AI Team"},
    {"Year": 2020, "Milestone": "GPT-3 Language Model Released", "Main_Persons": "OpenAI Team"},
    {"Year": 2020, "Milestone": "AlphaFold 2 Solves Protein Folding Problem", "Main_Persons": "DeepMind Team"},
    {"Year": 2022, "Milestone": "DALL-E 2 and Stable Diffusion (Text-to-Image Models)", "Main_Persons": "OpenAI Team, Stability AI Team"},
    {"Year": 2022, "Milestone": "ChatGPT Released", "Main_Persons": "OpenAI Team"},
    {"Year": 2023, "Milestone": "GPT-4 Released", "Main_Persons": "OpenAI Team"},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Sort the DataFrame by Year
df = df.sort_values(by="Year").reset_index(drop=True)

# Streamlit App
st.set_page_config(page_title="AI Milestones Timeline", layout="wide")

st.title("Significant AI Milestones Timeline")

# Filter milestones based on year range
st.sidebar.header("Filter Milestones")
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_range = st.sidebar.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

filtered_df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# Create a timeline chart using Plotly
fig = px.timeline(
    filtered_df,
    x_start="Year",
    x_end="Year",
    y="Milestone",
    hover_data=["Main_Persons"],
    labels={"Main_Persons": "Main Person(s)"},
)

fig.update_layout(
    yaxis={'categoryorder':'total ascending'},
    title="",
    xaxis_title="Year",
    yaxis_title="Milestone",
    hoverlabel_align="left",
    height=800,
)

# Adjust the x-axis to show integers only
fig.update_xaxes(dtick=1)

# Display the timeline
st.plotly_chart(fig, use_container_width=True)

# Show detailed information when a milestone is selected
st.header("Milestone Details")
selected_milestone = st.selectbox("Select a Milestone", filtered_df["Milestone"])

milestone_details = filtered_df[filtered_df["Milestone"] == selected_milestone].iloc[0]
st.markdown(f"**Year:** {milestone_details['Year']}")
st.markdown(f"**Milestone:** {milestone_details['Milestone']}")
st.markdown(f"**Main Person(s):** {milestone_details['Main_Persons']}")
