import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Streamlit App Configuration: Call this as the first command
st.set_page_config(page_title="AI Milestones Timeline", layout="wide")

# Custom CSS for dark gray background and white fonts, including sliders and other components
st.markdown(
    """
    <style>
    body {
        background-color: #2E2E2E;  /* Dark grey background */
        color: white;  /* White font color */
    }
    .css-18e3th9 {  /* Main frame */
        background-color: #2E2E2E;
        color: white;
    }
    .css-1aumxhk {  /* Sidebar */
        background-color: #2E2E2E;
        color: white;
    }
    .stSidebar {
        background-color: #2E2E2E;
        color: white;
    }
    .css-2trqyj {  /* Input text, dropdowns, and sliders */
        color: white !important;
    }
    .css-10trblm {  /* Slider values */
        color: white !important;
    }
    .css-1v3fvcr {  /* Main container */
        background-color: #2E2E2E;
        color: white;
    }
    .css-145kmo2 {  /* Title and header font color */
        color: white;
    }
    .css-1dp5vir {  /* Fix slider tick labels */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Data: List of AI Milestones with Descriptions
data = [
    {
        "Date": "1950-10-01",
        "Year": 1950,
        "Milestone": "Turing Test Proposed",
        "Main_Persons": "Alan Turing",
        "Description": "Alan Turing publishes 'Computing Machinery and Intelligence', introducing the concept of the Turing Test to determine if a machine can exhibit human-like intelligence."
    },
    {
        "Date": "1952-01-01",
        "Year": 1952,
        "Milestone": "First Self-Learning Program (Checkers)",
        "Main_Persons": "Arthur Samuel",
        "Description": "Arthur Samuel develops a checkers-playing program that learns and improves over time, demonstrating an early form of machine learning."
    },
    {
        "Date": "1956-06-18",
        "Year": 1956,
        "Milestone": "Dartmouth Conference (Birth of AI)",
        "Main_Persons": "John McCarthy, Marvin Minsky, Claude Shannon, Nathaniel Rochester",
        "Description": "The Dartmouth Conference is held, where the term 'Artificial Intelligence' is coined, marking the founding of AI as a field."
    },
    {
        "Date": "1956-01-01",
        "Year": 1956,
        "Milestone": "Logic Theorist (First AI Program)",
        "Main_Persons": "Allen Newell and Herbert A. Simon",
        "Description": "Allen Newell and Herbert A. Simon develop the Logic Theorist, considered the first AI program capable of proving mathematical theorems."
    },
    {
        "Date": "1957-01-01",
        "Year": 1957,
        "Milestone": "General Problem Solver",
        "Main_Persons": "Herbert A. Simon, J.C. Shaw, Allen Newell",
        "Description": "The General Problem Solver (GPS) is introduced as a universal problem-solving machine using heuristics."
    },
    {
        "Date": "1957-01-01",
        "Year": 1957,
        "Milestone": "Perceptron Invented",
        "Main_Persons": "Frank Rosenblatt",
        "Description": "Frank Rosenblatt invents the Perceptron, an early neural network capable of learning, laying the groundwork for modern neural networks."
    },
    {
        "Date": "1958-01-01",
        "Year": 1958,
        "Milestone": "LISP Programming Language",
        "Main_Persons": "John McCarthy",
        "Description": "John McCarthy develops LISP, a programming language designed for AI research and known for its excellent handling of symbolic information."
    },
    {
        "Date": "1966-01-01",
        "Year": 1966,
        "Milestone": "ELIZA (First Chatbot)",
        "Main_Persons": "Joseph Weizenbaum",
        "Description": "Joseph Weizenbaum creates ELIZA, a program that simulates conversation, demonstrating natural language processing capabilities."
    },
    {
        "Date": "1965-01-01",
        "Year": 1965,
        "Milestone": "Stanford AI Lab Founded",
        "Main_Persons": "John McCarthy",
        "Description": "John McCarthy establishes the Stanford Artificial Intelligence Laboratory (SAIL), contributing significantly to AI research."
    },
    {
        "Date": "1970-01-01",
        "Year": 1970,
        "Milestone": "First Robot Arm (Shakey)",
        "Main_Persons": "Charles Rosen, Nils Nilsson",
        "Description": "Shakey the robot is developed at SRI International, becoming the first mobile robot controlled by AI, capable of reasoning about its actions."
    },
    {
        "Date": "1970-01-01",
        "Year": 1970,
        "Milestone": "SHRDLU (Natural Language Understanding)",
        "Main_Persons": "Terry Winograd",
        "Description": "Terry Winograd develops SHRDLU, a program that could understand and respond to natural language commands in a micro-world."
    },
    {
        "Date": "1972-01-01",
        "Year": 1972,
        "Milestone": "MYCIN (Expert System for Medical Diagnosis)",
        "Main_Persons": "Edward Shortliffe",
        "Description": "Edward Shortliffe develops MYCIN, an expert system for diagnosing blood infections, pioneering rule-based AI systems in medicine."
    },
    {
        "Date": "1974-01-01",
        "Year": 1974,
        "Milestone": "Backpropagation Algorithm (Initial Development)",
        "Main_Persons": "Paul Werbos",
        "Description": "Paul Werbos introduces the backpropagation algorithm in his PhD thesis, enabling the training of multi-layer neural networks."
    },
    {
        "Date": "1980-01-01",
        "Year": 1980,
        "Milestone": "Expert Systems",
        "Main_Persons": "Edward Feigenbaum",
        "Description": "Edward Feigenbaum leads the development of expert systems, which emulate decision-making abilities of human experts."
    },
    {
        "Date": "1982-01-01",
        "Year": 1982,
        "Milestone": "Hopfield Network",
        "Main_Persons": "John Hopfield",
        "Description": "John Hopfield introduces Hopfield networks, a form of recurrent neural network with applications in associative memory."
    },
    {
        "Date": "1986-10-09",
        "Year": 1986,
        "Milestone": "Backpropagation Popularized",
        "Main_Persons": "David Rumelhart, Geoffrey Hinton, Ronald Williams",
        "Description": "Rumelhart, Hinton, and Williams publish papers popularizing backpropagation, revitalizing interest in neural networks."
    },
    {
        "Date": "1995-01-01",
        "Year": 1995,
        "Milestone": "Support Vector Machines Popularized",
        "Main_Persons": "Vladimir Vapnik",
        "Description": "Vladimir Vapnik and colleagues develop Support Vector Machines (SVMs), powerful for classification and regression tasks."
    },
    {
        "Date": "1997-05-11",
        "Year": 1997,
        "Milestone": "Deep Blue Defeats Garry Kasparov",
        "Main_Persons": "IBM Team led by Feng-hsiung Hsu",
        "Description": "IBM's Deep Blue defeats world chess champion Garry Kasparov, marking a milestone in AI's ability to challenge human intellect."
    },
    {
        "Date": "1997-01-01",
        "Year": 1997,
        "Milestone": "LSTM Networks Introduced",
        "Main_Persons": "Sepp Hochreiter and Jürgen Schmidhuber",
        "Description": "Hochreiter and Schmidhuber introduce LSTM networks, overcoming problems with training recurrent neural networks, especially for long sequences."
    },
    {
        "Date": "2002-09-01",
        "Year": 2002,
        "Milestone": "Roomba Released",
        "Main_Persons": "iRobot Corporation",
        "Description": "iRobot releases the Roomba, an autonomous robotic vacuum cleaner, bringing AI and robotics into consumer homes."
    },
    {
        "Date": "2006-10-02",
        "Year": 2006,
        "Milestone": "Netflix Prize Competition Begins",
        "Main_Persons": "Netflix",
        "Description": "Netflix launches a competition to improve its recommendation algorithm, spurring advances in collaborative filtering and machine learning."
    },
    {
        "Date": "2011-02-16",
        "Year": 2011,
        "Milestone": "IBM Watson Wins Jeopardy!",
        "Main_Persons": "IBM Team",
        "Description": "IBM's Watson defeats champions on the quiz show 'Jeopardy!', showcasing advances in natural language processing and AI."
    },
    {
        "Date": "2012-09-30",
        "Year": 2012,
        "Milestone": "AlexNet Wins ImageNet Competition",
        "Main_Persons": "Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton",
        "Description": "AlexNet achieves a dramatic improvement in image classification on ImageNet using deep convolutional neural networks, igniting the deep learning revolution."
    },
    {
        "Date": "2014-06-10",
        "Year": 2014,
        "Milestone": "Generative Adversarial Networks (GANs)",
        "Main_Persons": "Ian Goodfellow",
        "Description": "Ian Goodfellow introduces GANs, a framework where two neural networks contest with each other, leading to breakthroughs in generative modeling."
    },
    {
        "Date": "2016-03-12",
        "Year": 2016,
        "Milestone": "AlphaGo Defeats Lee Sedol",
        "Main_Persons": "DeepMind Team led by Demis Hassabis",
        "Description": "DeepMind's AlphaGo defeats world champion Lee Sedol in the game of Go, a major milestone in AI due to the game's complexity."
    },
    {
        "Date": "2017-06-12",
        "Year": 2017,
        "Milestone": "Transformer Model Introduced",
        "Main_Persons": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.",
        "Description": "The Transformer architecture is introduced, revolutionizing natural language processing by enabling models to handle sequential data without recurrent networks."
    },
    {
        "Date": "2018-10-11",
        "Year": 2018,
        "Milestone": "BERT Language Model",
        "Main_Persons": "Google AI Team",
        "Description": "Google releases BERT, setting new state-of-the-art results in NLP tasks through bidirectional training of Transformer models."
    },
    {
        "Date": "2020-05-28",
        "Year": 2020,
        "Milestone": "GPT-3 Released",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases GPT-3, a language model with 175 billion parameters, demonstrating remarkable capabilities in generating human-like text."
    },
    {
        "Date": "2020-11-30",
        "Year": 2020,
        "Milestone": "AlphaFold 2 Solves Protein Folding",
        "Main_Persons": "DeepMind Team",
        "Description": "DeepMind's AlphaFold 2 achieves a breakthrough in predicting protein structures, a significant advancement in biology and medicine."
    },
    {
        "Date": "2022-04-06",
        "Year": 2022,
        "Milestone": "DALL·E 2 Released",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases DALL·E 2, a model capable of generating images from textual descriptions, advancing generative AI."
    },
    {
        "Date": "2022-06-12",
        "Year": 2022,
        "Milestone": "Launch of Midjourney",
        "Main_Persons": "Midjourney, Inc",
        "Description": "Midjourney, an AI-powered image generation tool, was released to the public as a Discord bot, allowing users to create unique and imaginative images from text prompts."
    },
    {
        "Date": "2022-08-22",
        "Year": 2022,
        "Milestone": "Stable Diffusion Released",
        "Main_Persons": "Stability AI Team",
        "Description": "Stability AI releases Stable Diffusion, an open-source model capable of generating images from textual descriptions, advancing generative AI."
    },
    {
        "Date": "2022-11-30",
        "Year": 2022,
        "Milestone": "ChatGPT Released",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases ChatGPT, a conversational AI model that interacts in a dialogue format, capable of answering questions and engaging in discussions."
    },
    {
        "Date": "2023-03-14",
        "Year": 2023,
        "Milestone": "GPT-4 Released",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases GPT-4, an advanced language model with enhanced capabilities in understanding and generating text."
    },
    {
        "Date": "2023-05-10",
        "Year": 2023,
        "Milestone": "Google Introduces Gemini",
        "Main_Persons": "Google AI Team",
        "Description": "Google announces Gemini, a multimodal model suite with advanced conversational capabilities, integrated into Google Cloud for enterprise AI solutions."
    },
    {
        "Date": "2023-07-18",
        "Year": 2023,
        "Milestone": "Meta LLaMA 2 Launch",
        "Main_Persons": "Meta AI Team",
        "Description": "Meta releases LLaMA 2, an open-source large language model with 70 billion parameters, designed for research and commercial applications."
    },
    {
        "Date": "2023-01-01",
        "Year": 2023,
        "Milestone": "Cohere Command R",
        "Main_Persons": "Cohere AI Team",
        "Description": "Cohere launches Command R, a model optimized for retrieval-augmented generation, tailored for enterprises dealing with large datasets."
    },
    {
        "Date": "2023-03-14",
        "Year": 2023,
        "Milestone": "Claude Released",
        "Main_Persons": "Anthropic",
        "Description": "Claude is a family of large language models developed by Anthropic, based on Constitutional AI."
    },
    {
        "Date": "2023-09-01",
        "Year": 2023,
        "Milestone": "Snowflake Arctic and Partnership with Mistral AI",
        "Main_Persons": "Snowflake AI Team",
        "Description": "Snowflake introduces Arctic, an open-source language model, in partnership with Mistral AI to enhance retrieval-augmented generation for enterprise use."
    },
    {
        "Date": "2023-09-01",
        "Year": 2023,
        "Milestone": "Databricks Mosaic and Integration of Mistral Models",
        "Main_Persons": "Databricks AI Team",
        "Description": "Databricks releases Mosaic, a platform for deploying and fine-tuning foundation models, integrating Mistral’s AI models for enterprise data processing."
    },
    {
        "Date": "2023-03-21",
        "Year": 2023,
        "Milestone": "NVIDIA NeMo Framework Expansion",
        "Main_Persons": "NVIDIA AI Team",
        "Description": "NVIDIA expands its NeMo framework to support multimodal AI models and high-performance computing, catering to industries needing fast AI model development."
    },
    {
        "Date": "2024-04-18",
        "Year": 2024,
        "Milestone": "Meta LLaMA 3 Release",
        "Main_Persons": "Meta AI Team",
        "Description": "Meta releases LLaMA 3, an open-source model with up to 70 billion parameters, showcasing enhanced reasoning and multilingual capabilities."
    },
    {
        "Date": "2024-09-12",
        "Year": 2024,
        "Milestone": "OpenAI O1 Release",
        "Main_Persons": "OpenAI Team",
        "Description": "OpenAI releases the O1 model series, trained with large-scale reinforcement learning to reason using chain of thought."
    }
]

# Create a DataFrame
df = pd.DataFrame(data)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by Date
df = df.sort_values(by="Date").reset_index(drop=True)

# Calculate cumulative milestones
df["Cumulative"] = range(1, len(df) + 1)

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
    step=1
)
filtered_df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])].reset_index(drop=True)

# Milestone selection from the sidebar
st.sidebar.header("Select a Milestone")
selected_milestone_sidebar = st.sidebar.selectbox(
    "Choose a Milestone", ["None"] + filtered_df["Milestone"].tolist()
)

# Generate marker colors based on selection
marker_colors = [
    "red" if milestone == selected_milestone_sidebar else "lightblue"
    for milestone in filtered_df["Milestone"]
]

# Create cumulative milestones line chart with a dark mode theme
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Cumulative"],
        mode="lines+markers",
        marker=dict(size=8, color=marker_colors),
        line=dict(color="cyan"),
        hovertext=filtered_df.apply(
            lambda row: f"<b>Date:</b> {row['Date'].strftime('%d %B, %Y')}<br>"
                        f"<b>Milestone:</b> {row['Milestone']}",
            axis=1,
        ),
        hoverinfo="text",
    )
)

# Apply dark mode settings
fig.update_layout(
    paper_bgcolor="black",  # Background of the entire chart
    plot_bgcolor="black",  # Background of the plot area
    font_color="white",  # Font color for labels
    height=500,
    showlegend=False,
    xaxis_title="Date",
    yaxis_title="Cumulative Milestones",
    xaxis=dict(
        range=[
            pd.Timestamp(year=year_range[0], month=1, day=1) - pd.DateOffset(months=6),
            pd.Timestamp(year=year_range[1], month=12, day=31) + pd.DateOffset(months=6),
        ],
        tickformat="%B %Y",
        tickmode="auto",
        nticks=20,
        color="white"  # White tick labels
    ),
    yaxis=dict(
        range=[0, filtered_df["Cumulative"].max() + 1],
        color="white"  # White tick labels
    ),
    hovermode="closest",
)

# Display the cumulative milestones line chart
st.plotly_chart(fig, use_container_width=True)

# Show detailed information when a milestone is selected
st.header("Milestone Details")

if selected_milestone_sidebar != "None":
    milestone_details = filtered_df[filtered_df["Milestone"] == selected_milestone_sidebar].iloc[0]
    st.markdown(f"**Date:** {milestone_details['Date'].strftime('%d %B, %Y')}")
    st.markdown(f"**Milestone:** {milestone_details['Milestone']}")
    st.markdown(f"**Main Person(s):** {milestone_details['Main_Persons']}")
    st.markdown(f"**Description:** {milestone_details['Description']}")
else:
    st.write("Select a milestone from the sidebar to see the details.")
