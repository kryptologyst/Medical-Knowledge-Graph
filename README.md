# Medical Knowledge Graph - Research Demo

**DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use or medical diagnosis. This software is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.**

## Overview

This project implements a modern Medical Knowledge Graph (KG) system for clinical decision support and reasoning. It provides:

- Knowledge graph construction and embedding learning
- Link prediction and reasoning capabilities  
- Clinical decision support queries
- Explainable AI for medical reasoning
- Interactive demos for exploration

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```

2. **Run the interactive demo:**
   ```bash
   streamlit run demo/app.py
   ```

3. **Train a knowledge graph model:**
   ```bash
   python scripts/train.py --config configs/baseline.yaml
   ```

## Dataset Schema

The system works with medical knowledge graphs in the following format:

- **Entities**: Diseases, drugs, symptoms, treatments, procedures
- **Relations**: treated_by, side_effect, causes, contraindicated_with, etc.
- **Format**: Triples (head, relation, tail) in CSV/JSON format

### Synthetic Dataset

A synthetic medical knowledge graph is included for demonstration purposes, containing:
- 1,000+ medical entities
- 5,000+ relationships
- Multiple relation types
- Realistic medical hierarchies

### Real-world Integration

The system can integrate with:
- UMLS (Unified Medical Language System)
- SNOMED CT
- DrugBank
- MeSH (Medical Subject Headings)

## Training and Evaluation

### Training Commands

```bash
# Train TransE model
python scripts/train.py --config configs/transe.yaml

# Train RotatE model  
python scripts/train.py --config configs/rotate.yaml

# Train ComplEx model
python scripts/train.py --config configs/complex.yaml
```

### Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct answers
- **Hits@1/3/10**: Percentage of correct answers in top-1/3/10 predictions
- **Filtered Metrics**: Exclude known true triples during evaluation
- **Path-based Reasoning**: Multi-hop reasoning accuracy

## Demo Features

The Streamlit demo provides:

1. **Graph Visualization**: Interactive network visualization
2. **Query Interface**: Natural language queries about medical relationships
3. **Link Prediction**: Predict missing relationships
4. **Explainability**: Show reasoning paths and confidence scores
5. **Entity Search**: Find related medical concepts

## Model Architecture

### Knowledge Graph Embeddings

- **TransE**: Translational embedding model
- **RotatE**: Rotational embedding model  
- **ComplEx**: Complex-valued embeddings
- **Graph Neural Networks**: Message passing for reasoning

### Reasoning Methods

- **Path-based**: Multi-hop reasoning along graph paths
- **Embedding-based**: Vector similarity for link prediction
- **Hybrid**: Combine symbolic and neural approaches

## Configuration

Models are configured via YAML files in `configs/`:

```yaml
model:
  name: "TransE"
  embedding_dim: 100
  margin: 1.0

training:
  epochs: 100
  batch_size: 512
  learning_rate: 0.001

data:
  train_path: "data/synthetic_medical_kg.csv"
  valid_path: "data/valid_medical_kg.csv"
```

## Known Limitations

- Synthetic data may not reflect real-world medical complexity
- Limited to the specific medical domains in the training data
- Requires domain expert validation for clinical applications
- Performance may vary across different medical specialties

## Safety and Compliance

- **No PHI/PII**: System does not process personal health information
- **Research Only**: Not validated for clinical decision making
- **Transparency**: All model decisions are explainable
- **Bias Awareness**: Includes fairness evaluation across medical domains

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{medical_knowledge_graph,
  title={Medical Knowledge Graph for Clinical Decision Support},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Medical-Knowledge-Graph}
}
```
# Medical-Knowledge-Graph
