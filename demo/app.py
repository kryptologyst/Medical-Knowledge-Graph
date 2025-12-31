"""Streamlit demo for Medical Knowledge Graph."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
import json
import torch
from typing import Dict, List, Tuple, Optional

# Set page config
st.set_page_config(
    page_title="Medical Knowledge Graph Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add disclaimer banner
st.error("""
**DISCLAIMER: This is a research demonstration and is NOT intended for clinical use or medical diagnosis. 
This software is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.**
""")

# Title
st.title("🏥 Medical Knowledge Graph Demo")
st.markdown("Interactive exploration of medical knowledge graphs for clinical decision support and reasoning.")

# Sidebar
st.sidebar.title("Configuration")

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample medical knowledge graph data."""
    # Create sample medical triples
    sample_triples = [
        ("hypertension", "treated_by", "amlodipine"),
        ("hypertension", "treated_by", "lisinopril"),
        ("diabetes", "treated_by", "metformin"),
        ("diabetes", "treated_by", "insulin"),
        ("metformin", "side_effect", "nausea"),
        ("metformin", "side_effect", "diarrhea"),
        ("amlodipine", "side_effect", "dizziness"),
        ("amlodipine", "side_effect", "swelling"),
        ("insulin", "side_effect", "hypoglycemia"),
        ("heart_disease", "causes", "chest_pain"),
        ("heart_disease", "causes", "shortness_of_breath"),
        ("pneumonia", "symptom_of", "fever"),
        ("pneumonia", "symptom_of", "cough"),
        ("warfarin", "contraindicated_with", "aspirin"),
        ("warfarin", "contraindicated_with", "ibuprofen"),
        ("surgery", "performed_on", "heart"),
        ("surgery", "performed_on", "lung"),
        ("chemotherapy", "treated_by", "cancer"),
        ("radiation_therapy", "treated_by", "cancer"),
        ("aspirin", "prevents", "heart_attack"),
        ("aspirin", "prevents", "stroke"),
        ("exercise", "reduces", "hypertension"),
        ("exercise", "reduces", "diabetes"),
        ("smoking", "increases_risk_of", "lung_cancer"),
        ("smoking", "increases_risk_of", "heart_disease"),
    ]
    
    return sample_triples

# Load data
triples = load_sample_data()

# Create entity and relation mappings
entities = set()
relations = set()
for head, relation, tail in triples:
    entities.add(head)
    entities.add(tail)
    relations.add(relation)

entities = sorted(list(entities))
relations = sorted(list(relations))

# Sidebar controls
st.sidebar.subheader("Graph Visualization")
show_labels = st.sidebar.checkbox("Show Labels", value=True)
layout_type = st.sidebar.selectbox("Layout", ["spring", "circular", "hierarchical"])

st.sidebar.subheader("Query Interface")
query_type = st.sidebar.selectbox("Query Type", [
    "Find treatments for a disease",
    "Find side effects of a drug", 
    "Find diseases with a symptom",
    "Find drug interactions",
    "Find risk factors"
])

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Knowledge Graph Visualization")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    for head, relation, tail in triples:
        G.add_edge(head, tail, relation=relation)
    
    # Choose layout
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    else:  # hierarchical
        pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Create Plotly network visualization
    edge_x = []
    edge_y = []
    edge_labels = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Get relation label
        relation = G[edge[0]][edge[1]]['relation']
        edge_labels.append(relation)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Create hover text with connections
        neighbors = list(G.neighbors(node))
        hover_text = f"<b>{node}</b><br>Connections: {len(neighbors)}"
        node_hovertext.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        hoverinfo='text',
        hovertext=node_hovertext,
        text=node_text if show_labels else [],
        textposition="middle center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2, color='black')
        )
    )
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Medical Knowledge Graph',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive medical knowledge graph showing relationships between diseases, drugs, symptoms, and treatments",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='gray', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Query Interface")
    
    # Query interface based on selected type
    if query_type == "Find treatments for a disease":
        selected_disease = st.selectbox("Select Disease", entities)
        
        # Find treatments
        treatments = []
        for head, relation, tail in triples:
            if head == selected_disease and relation == "treated_by":
                treatments.append(tail)
        
        if treatments:
            st.success(f"**Treatments for {selected_disease}:**")
            for treatment in treatments:
                st.write(f"• {treatment}")
        else:
            st.info(f"No treatments found for {selected_disease}")
    
    elif query_type == "Find side effects of a drug":
        selected_drug = st.selectbox("Select Drug", entities)
        
        # Find side effects
        side_effects = []
        for head, relation, tail in triples:
            if head == selected_drug and relation == "side_effect":
                side_effects.append(tail)
        
        if side_effects:
            st.warning(f"**Side effects of {selected_drug}:**")
            for effect in side_effects:
                st.write(f"• {effect}")
        else:
            st.info(f"No side effects found for {selected_drug}")
    
    elif query_type == "Find diseases with a symptom":
        selected_symptom = st.selectbox("Select Symptom", entities)
        
        # Find diseases
        diseases = []
        for head, relation, tail in triples:
            if tail == selected_symptom and relation in ["symptom_of", "causes"]:
                diseases.append(head)
        
        if diseases:
            st.info(f"**Diseases associated with {selected_symptom}:**")
            for disease in diseases:
                st.write(f"• {disease}")
        else:
            st.info(f"No diseases found for {selected_symptom}")
    
    elif query_type == "Find drug interactions":
        selected_drug = st.selectbox("Select Drug", entities)
        
        # Find interactions
        interactions = []
        for head, relation, tail in triples:
            if head == selected_drug and relation == "contraindicated_with":
                interactions.append(tail)
        
        if interactions:
            st.error(f"**Drug interactions for {selected_drug}:**")
            for interaction in interactions:
                st.write(f"• {interaction}")
        else:
            st.info(f"No interactions found for {selected_drug}")
    
    elif query_type == "Find risk factors":
        selected_condition = st.selectbox("Select Condition", entities)
        
        # Find risk factors
        risk_factors = []
        for head, relation, tail in triples:
            if tail == selected_condition and relation == "increases_risk_of":
                risk_factors.append(head)
        
        if risk_factors:
            st.warning(f"**Risk factors for {selected_condition}:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.info(f"No risk factors found for {selected_condition}")

# Statistics section
st.subheader("Knowledge Graph Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Entities", len(entities))

with col2:
    st.metric("Total Relations", len(relations))

with col3:
    st.metric("Total Triples", len(triples))

with col4:
    avg_connections = np.mean([len(list(G.neighbors(node))) for node in G.nodes()])
    st.metric("Avg Connections", f"{avg_connections:.1f}")

# Entity type distribution
st.subheader("Entity Type Distribution")

entity_types = {
    "Diseases": ["hypertension", "diabetes", "heart_disease", "pneumonia", "cancer", "lung_cancer"],
    "Drugs": ["amlodipine", "metformin", "insulin", "warfarin", "aspirin", "ibuprofen"],
    "Symptoms": ["chest_pain", "fever", "cough", "nausea", "dizziness", "swelling"],
    "Treatments": ["surgery", "chemotherapy", "radiation_therapy", "exercise"],
    "Risk Factors": ["smoking"]
}

type_counts = {}
for entity_type, type_entities in entity_types.items():
    count = sum(1 for entity in entities if entity in type_entities)
    type_counts[entity_type] = count

# Add "Other" category
other_count = len(entities) - sum(type_counts.values())
if other_count > 0:
    type_counts["Other"] = other_count

# Create pie chart
fig_pie = px.pie(
    values=list(type_counts.values()),
    names=list(type_counts.keys()),
    title="Entity Type Distribution"
)
st.plotly_chart(fig_pie, use_container_width=True)

# Relation frequency
st.subheader("Relation Frequency")

relation_counts = {}
for _, relation, _ in triples:
    relation_counts[relation] = relation_counts.get(relation, 0) + 1

# Create bar chart
fig_bar = px.bar(
    x=list(relation_counts.keys()),
    y=list(relation_counts.values()),
    title="Relation Frequency",
    labels={'x': 'Relation Type', 'y': 'Count'}
)
fig_bar.update_xaxis(tickangle=45)
st.plotly_chart(fig_bar, use_container_width=True)

# Raw data section
st.subheader("Raw Knowledge Graph Data")

# Create DataFrame
df = pd.DataFrame(triples, columns=['Head', 'Relation', 'Tail'])
st.dataframe(df, use_container_width=True)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="medical_knowledge_graph.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("""
**About this Demo:**
- This is a research demonstration of medical knowledge graph capabilities
- The data shown is synthetic and for educational purposes only
- Real-world medical knowledge graphs would be much larger and more complex
- Always consult healthcare professionals for medical advice
""")
