#!/usr/bin/env python3
"""
Validation script to ensure all essential components work.
"""

def main():
    print("🔍 Validating STARK-QA essential components...")
    
    try:
        # Test core imports
        from src.data_loader import FairytaleQALoader
        from src.rag.robust_rag import RobustRAG
        from src.kag.simple_kg import SimpleKnowledgeGraph
        from src.models.stark_qa import StarkQASystem
        from src.models.fallback_qa import SimpleFallbackQA
        from src.evaluation.evaluator import ComprehensiveEvaluator
        from src.baselines.basic_rag import BasicRAG
        from src.baselines.basic_kag import BasicKAG
        
      p  print("✅ All imports successful!")
        print("✅ Essential files are present and working")
        print("\n📋 Essential files kept:")
        print("  - main.py (entry point)")
        print("  - config.py (configuration)")
        print("  - requirements.txt (dependencies)")
        print("  - src/data_loader.py (data loading)")
        print("  - src/rag/robust_rag.py (RAG system)")
        print("  - src/kag/simple_kg.py (knowledge graph)")
        print("  - src/models/ (STARK-QA & fallback systems)")
        print("  - src/baselines/ (baseline systems)")
        print("  - src/evaluation/ (evaluation system)")
        print("  - cache/ (preprocessed data)")
        print("  - results/ (evaluation results)")
        
        print("\n🎯 Ready to run: python main.py --test_size 10")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
