import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
    {
        "Year": 1956,
        "Milestone": "Logic Theorist (First AI Program)",
        "Main_Persons": "Allen Newell and Herbert A. Simon",
        "Description": "Allen Newell and Herbert A. Simon develop the Logic Theorist, considered the first AI program capable of proving mathematical theorems."
    },
    {
        "Year": 1957,
        "Milestone": "General Problem Solver",
        "Main_Persons": "Herbert A. Simon, J.C. Shaw, Allen Newell",
        "Description": "The General Problem Solver (GPS) is introduced as a universal problem-solving machine using heuristics."
    },
    {
        "Year": 1957,
        "Milestone": "Perceptron Invented",
        "Main_Persons": "Frank Rosenblatt",
        "Description": "Frank Rosenblatt invents the Perceptron, an early neural network capable of learning, laying the groundwork for modern neural networks."
    },
    {
        "Year": 1958,
        "Milestone": "LISP Programming Language",
        "Main_Persons": "John McCarthy",
        "Description": "John McCarthy develops LISP, a programming language designed for AI research and known for its excellent handling of symbolic information."
    },
    {
        "Year": 1965,
        "Milestone": "ELIZA (First Chatbot)",
        "Main_Persons": "Joseph Weizenbaum",
        "Description": "Joseph Weizenbaum creates ELIZA, a program that simulates conversation, demonstrating natural language processing capabilities."
    },
    {
        "Year": 1969,
        "Milestone": "Stanford AI Lab Founded",
        "Main_Persons": "John McCarthy",
        "Description": "John McCarthy establishes the Stanford Artificial Intelligence Laboratory (SAIL), contributing significantly to AI research."
    },
    {
        "Year": 1970,
        "Milestone": "First Robot Arm (Shakey)",
        "Main_Persons": "Charles Rosen, Nils Nilsson",
        "Description": "Shakey the robot is developed at SRI International, becoming the first mobile robot controlled by AI, capable of reasoning about its actions."
    },
    {
        "Year": 1970,
        "Milestone": "SHRDLU (Natural Language Understanding)",
        "Main_Persons": "Terry Winograd",
        "Description": "Terry Winograd develops SHRDLU, a program that could understand and respond to natural language commands in a micro-world."
    },
    {
        "Year": 1972,
        "Milestone": "MYCIN (Expert System for Medical Diagnosis)",
        "Main_Persons": "Edward Shortliffe",
        "Description": "Edward Shortliffe develops MYCIN, an expert system for diagnosing blood infections, pioneering rule-based AI systems in medicine."
    },
    {
        "Year": 1974,
        "Milestone": "Backpropagation Algorithm (Initial Development)",
        "Main_Persons": "Paul Werbos",
        "Description": "Paul Werbos introduces the backpropagation algorithm in his PhD thesis, enabling the training of multi-layer neural networks."
    },
    {
        "Year": 1980,
        "Milestone": "Expert Systems",
        "Main_Persons": "Edward Feigenbaum",
        "Description": "Edward Feigenbaum leads the development of expert systems, which emulate decision-making abilities of human experts."
    },
    {
        "Year": 1982,
        "Milestone": "Hopfield Network",
        "Main_Persons": "John Hopfield",
        "Description": "John Hopfield introduces Hopfield networks, a form of recurrent neural network with applications in associative memory."
    },
    {
        "Year": 1986,
        "Milestone": "Backpropagation Popularized",
        "Main_Persons": "David Rumelhart, Geoffrey Hinton, Ronald Williams",
        "Description": "Rumelhart, Hinton, and Williams publish papers popularizing backpropagation, revitalizing interest in neural networks."
    },
    {
        "Year": 1995,
        "Milestone": "Support Vector Machines Popularized",
        "Main_Persons": "Vladimir Vapnik",
        "Description": "Vladimir Vapnik and colleagues develop Support Vector Machines (SVMs), powerful for classification and regression tasks."
    },
    {
        "Year": 1997,
        "Milestone": "Deep Blue Defeats Garry Kasparov",
        "Main_Persons": "IBM Team led by Feng-hsiung Hsu",
        "Description": "IBM's Deep Blue defeats world chess champion Garry Kasparov, marking a milestone in AI's ability to challenge human intellect."
    },
    {
        "Year": 1997,
        "Milestone": "LSTM Networks Introduced",
        "Main_Persons": "Sepp Hochreiter and Jürgen Schmidhuber",
        "Description": "Hochreiter and Schmidhuber introduce LSTM networks, overcoming problems with training recurrent neural networks, especially for long sequences."
    },
    {
        "Year": 2002,
        "Milestone": "Roomba Released",
        "Main_Persons": "iRobot Corporation",
        "Description": "iRobot releases the Roomba, an autonomous robotic vacuum cleaner, bringing AI and robotics into consumer homes."
    },
    {
        "Year": 2006,
        "Milestone": "Netflix Prize Competition Begins",
        "Main_Persons": "Netflix",
        "Description": "Netflix launches a competition to improve its recommendation algorithm, spurring advances in collaborative filtering and machine learning."
    },
    {
        "Year": 2011,
        "Milestone": "IBM Watson Wins Jeopardy!",
        "Main_Persons": "IBM Team",
        "Description": "IBM's Watson defeats champions on the quiz show 'Jeopardy!', showcasing advances in natural language processing and AI."
    },
    {
        "Year": 2012,
        "Milestone": "AlexNet Wins ImageNet Competition",
        "Main_Persons": "Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton",
        "Description": "AlexNet achieves a dramatic improvement in image classification on ImageNet using deep convolutional neural networks, igniting the deep learning revolution."
    },
    {
        "Year": 2014,
        "Milestone": "Generative Adversarial Networks (GANs)",
        "Main_Persons": "Ian Goodfellow",
        "Description": "Ian Goodfellow introduces GANs, a framework where two neural networks contest with each other, leading to breakthroughs in generative modeling."
    },
    {
        "Year": 2016,
        "Milestone": "AlphaGo Defeats Lee Sedol",
        "Main_Persons": "DeepMind Team led by Demis Hassabis",
        "Description": "DeepMind's AlphaGo defeats world champion Lee Sedol in the game of Go, a major milestone in AI due to the game's complexity."
    },
    {
        "Year": 2017,
        "Milestone": "Transformer Model Introduced",
        "Main_Persons": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.",
        "Description": "The Transformer architecture is introduced, revolutionizing natural language processing by enabling models to handle sequential data without recurrent networks."
    },
    {
        "Year": 2018,
        "Milestone": "BERT Language Model",
        "Main_Persons": "Google AI Team",
        "Description": "Google releases BERT, setting new state-of-the-art results in NLP tasks through bidirectional training of Transformer models."
    },
    {
        "Year": 2020,
        "Milestone": "GPT-3 Released",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases GPT-3, a language model with 175 billion parameters, demonstrating remarkable capabilities in generating human-like text."
    },
    {
        "Year": 2020,
        "Milestone": "AlphaFold 2 Solves Protein Folding",
        "Main_Persons": "DeepMind Team",
        "Description": "DeepMind's AlphaFold 2 achieves a breakthrough in predicting protein structures, a significant advancement in biology and medicine."
    },
    {
        "Year": 2022,
        "Milestone": "DALL·E 2 and Stable Diffusion",
        "Main_Persons": "OpenAI Team, Stability AI Team",
        "Description": "OpenAI and Stability AI release models capable of generating images from textual descriptions, advancing generative AI."
    },
    {
        "Year": 2022,
        "Milestone": "ChatGPT Released",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases ChatGPT, a conversational AI model that interacts in a dialogue format, capable of answering questions and engaging in discussions."
    },
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

# Milestone selection
st.sidebar.header("Select a Milestone")
selected_milestone = st.sidebar.selectbox("Choose a Milestone", ["None"] + filtered_df["Milestone"].tolist())

# Create cumulative milestones line chart
fig = go.Figure()

# Highlight the selected milestone
if selected_milestone != "None":
    selected_year = filtered_df[filtered_df["Milestone"] == selected_milestone]["Year"].values[0]
    fig.add_trace(
        go.Scatter(
            x=filtered_df["Year"],
            y=filtered_df["Cumulative"],
            mode="lines+markers",
            marker=dict(
                size=8,
                color=["red" if year == selected_year else "DarkBlue" for year in filtered_df["Year"]],
            ),
            line=dict(color="DarkBlue"),
            hovertext=filtered_df.apply(
                lambda row: f"<b>Year:</b> {row['Year']}<br>"
                            f"<b>Milestone:</b> {row['Milestone']}",
                axis=1
            ),
            hoverinfo="text",
        )
    )
else:
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

# Display the cumulative milestones line chart
st.plotly_chart(fig, use_container_width=True)

# Show detailed information when a milestone is selected
st.header("Milestone Details")

if selected_milestone != "None":
    milestone_details = filtered_df[filtered_df["Milestone"] == selected_milestone].iloc[0]
    st.markdown(f"**Year:** {milestone_details['Year']}")
    st.markdown(f"**Milestone:** {milestone_details['Milestone']}")
    st.markdown(f"**Main Person(s):** {milestone_details['Main_Persons']}")
    st.markdown(f"**Description:** {milestone_details['Description']}")
else:
    st.write("Select a milestone from the sidebar to see the details.")


