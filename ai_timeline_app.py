import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# AI milestones data
data = [
    {"Year": 1950, "Milestone": "Turing Test Proposed", "Person": "Alan Turing"},
    {"Year": 1952, "Milestone": "First Self-Learning Program (Checkers)", "Person": "Arthur Samuel"},
    {"Year": 1956, "Milestone": "Dartmouth Conference (Birth of AI)", "Person": "John McCarthy et al."},
    {"Year": 1956, "Milestone": "Logic Theorist (First AI Program)", "Person": "Allen Newell and Herbert A. Simon"},
    {"Year": 1957, "Milestone": "General Problem Solver", "Person": "Herbert A. Simon et al."},
    {"Year": 1957, "Milestone": "Perceptron Invented", "Person": "Frank Rosenblatt"},
    {"Year": 1958, "Milestone": "LISP programming language", "Person": "John McCarthy"},
    {"Year": 1965, "Milestone": "ELIZA (first chatbot)", "Person": "Joseph Weizenbaum"},
    {"Year": 1969, "Milestone": "Stanford AI Lab founded", "Person": "John McCarthy"},
    {"Year": 1970, "Milestone": "First robot arm (Shakey)", "Person": "Charles Rosen, Nils Nilsson"},
    {"Year": 1970, "Milestone": "SHRDLU (Natural Language Understanding)", "Person": "Terry Winograd"},
    {"Year": 1972, "Milestone": "MYCIN (Expert System for Medical Diagnosis)", "Person": "Edward Shortliffe"},
    {"Year": 1974, "Milestone": "Backpropagation Algorithm in Neural Networks (initial)", "Person": "Paul Werbos"},
    {"Year": 1980, "Milestone": "Expert systems", "Person": "Edward Feigenbaum"},
    {"Year": 1982, "Milestone": "Hopfield Network", "Person": "John Hopfield"},
    {"Year": 1986, "Milestone": "Backpropagation for neural networks (popularized)", "Person": "David Rumelhart et al."},
    {"Year": 1995, "Milestone": "Support Vector Machines Popularized", "Person": "Vladimir Vapnik"},
    {"Year": 1997, "Milestone": "IBM's Deep Blue defeats world chess champion", "Person": "IBM team"},
    {"Year": 1997, "Milestone": "Long Short-Term Memory (LSTM) Networks Introduced", "Person": "Hochreiter & Schmidhuber"},
    {"Year": 2002, "Milestone": "Roomba (first mass-market autonomous robot)", "Person": "iRobot Corporation"},
    {"Year": 2006, "Milestone": "Netflix Prize competition begins", "Person": "Netflix"},
    {"Year": 2011, "Milestone": "IBM Watson wins Jeopardy!", "Person": "IBM team"},
    {"Year": 2012, "Milestone": "ImageNet breakthrough (AlexNet)", "Person": "Krizhevsky, Sutskever, Hinton"},
    {"Year": 2014, "Milestone": "Generative Adversarial Networks (GANs)", "Person": "Ian Goodfellow"},
    {"Year": 2016, "Milestone": "AlphaGo defeats world Go champion", "Person": "DeepMind team"},
    {"Year": 2017, "Milestone": "Transformer Model Introduced", "Person": "Vaswani et al."},
    {"Year": 2018, "Milestone": "BERT language model", "Person": "Google AI team"},
    {"Year": 2020, "Milestone": "GPT-3 language model released", "Person": "OpenAI team"},
    {"Year": 2020, "Milestone": "AlphaFold 2 Solves Protein Folding Problem", "Person": "DeepMind Team"},
    {"Year": 2022, "Milestone": "DALL-E 2 and Stable Diffusion", "Person": "OpenAI team, Stability AI"},
    {"Year": 2022, "Milestone": "ChatGPT released", "Person": "OpenAI team"},
    {"Year": 2023, "Milestone": "GPT-4 released", "Person": "OpenAI team"},
]
