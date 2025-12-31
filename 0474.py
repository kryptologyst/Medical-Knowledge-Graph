#!/usr/bin/env python3
"""
Project 474: Medical Knowledge Graph - Modernized Example

This is a simple demonstration of the modernized medical knowledge graph system.
For the full implementation, see the src/ directory and run the Streamlit demo.

DISCLAIMER: This is a research demonstration and is NOT intended for clinical use.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.dataset import create_synthetic_medical_kg
from src.models.embeddings import TransE
from src.utils.core import set_seed, get_device, setup_logging
import torch


def main():
    """Demonstrate the modernized medical knowledge graph system."""
    print("🏥 Medical Knowledge Graph - Modernized Demo")
    print("=" * 50)
    
    # Setup
    set_seed(42)
    logger = setup_logging()
    device = get_device()
    
    print(f"Using device: {device}")
    print()
    
    # Generate synthetic medical knowledge graph
    print("Generating synthetic medical knowledge graph...")
    triples = create_synthetic_medical_kg(
        num_entities=100,
        num_relations=15,
        num_triples=200
    )
    
    print(f"Generated {len(triples)} medical triples")
    print()
    
    # Show some example triples
    print("Sample medical relationships:")
    for i, (head, relation, tail) in enumerate(triples[:5]):
        print(f"  {i+1}. {head} --[{relation}]--> {tail}")
    print()
    
    # Create entity and relation mappings
    entities = set()
    relations = set()
    for head, relation, tail in triples:
        entities.add(head)
        entities.add(tail)
        relations.add(relation)
    
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
    relation_to_id = {relation: idx for idx, relation in enumerate(sorted(relations))}
    
    print(f"Knowledge Graph Statistics:")
    print(f"  - Entities: {len(entities)}")
    print(f"  - Relations: {len(relations)}")
    print(f"  - Triples: {len(triples)}")
    print()
    
    # Demonstrate TransE model
    print("Initializing TransE embedding model...")
    model = TransE(
        num_entities=len(entities),
        num_relations=len(relations),
        embedding_dim=50,
        margin=1.0
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Simple query demonstration
    print("Query Examples:")
    
    # Find treatments for hypertension
    hypertension_treatments = [
        tail for head, relation, tail in triples 
        if head == "hypertension" and relation == "treated_by"
    ]
    
    if hypertension_treatments:
        print(f"  Treatments for hypertension: {', '.join(hypertension_treatments)}")
    else:
        print("  No treatments found for hypertension in synthetic data")
    
    # Find side effects of metformin
    metformin_side_effects = [
        tail for head, relation, tail in triples 
        if head == "metformin" and relation == "side_effect"
    ]
    
    if metformin_side_effects:
        print(f"  Side effects of metformin: {', '.join(metformin_side_effects)}")
    else:
        print("  No side effects found for metformin in synthetic data")
    
    print()
    
    # Demonstrate model prediction
    print("Model Prediction Example:")
    if "hypertension" in entity_to_id and "treated_by" in relation_to_id:
        head_id = entity_to_id["hypertension"]
        relation_id = relation_to_id["treated_by"]
        
        head_tensor = torch.tensor([head_id], device=device)
        relation_tensor = torch.tensor([relation_id], device=device)
        
        # Get predictions (this would be trained in the full system)
        with torch.no_grad():
            predictions = model.predict(head_tensor, relation_tensor)
            top_predictions = torch.topk(predictions, k=3, dim=1)
        
        print(f"  Top 3 predicted treatments for hypertension:")
        id_to_entity = {v: k for k, v in entity_to_id.items()}
        for i, idx in enumerate(top_predictions.indices[0]):
            entity_name = id_to_entity[idx.item()]
            score = top_predictions.values[0][i].item()
            print(f"    {i+1}. {entity_name} (score: {score:.3f})")
    
    print()
    print("✅ Demo completed!")
    print()
    print("Next Steps:")
    print("  1. Run the full training: python scripts/train.py --config configs/baseline.yaml")
    print("  2. Launch the interactive demo: streamlit run demo/app.py")
    print("  3. Explore the source code in src/ directory")
    print()
    print("⚠️  Remember: This is for research and education only, not clinical use!")


if __name__ == "__main__":
    main()
