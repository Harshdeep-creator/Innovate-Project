import argparse
import sys
import os
import yaml
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from src.pathway_pipeline import PathwayPipeline
from src.narrative_analyzer import NarrativeAnalyzer
from src.consistency_checker import ConsistencyChecker
from src.evidence_extractor import EvidenceExtractor
from src.utils import load_config, validate_inputs, save_results

class NarrativeConsistencySystem:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.pathway_pipeline = None
        self.analyzer = None
        self.checker = None
        self.extractor = None
        self.results = []
        
        directories = ["output", "data/novels", "evidence_dossiers", "logs", "models", ".cache"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_components(self, mode="accurate"):
        logger.info("Initializing system components...")
        
        self.pathway_pipeline = PathwayPipeline(
            config=self.config.get('pathway', {}),
            mode=mode
        )
        
        self.analyzer = NarrativeAnalyzer(
            config=self.config.get('models', {}),
            narrative_knowledge=self.config.get('narratives', {})
        )
        
        self.checker = ConsistencyChecker(
            config=self.config.get('analysis', {}),
            mode=mode
        )
        
        self.extractor = EvidenceExtractor(
            config=self.config.get('processing', {})
        )
        
        logger.info("All components initialized successfully")
    
    def process_backstory(self, backstory_data):
        try:
            story_id = backstory_data.get('id')
            book_name = backstory_data.get('book_name')
            character = backstory_data.get('char')
            content = backstory_data.get('content', '')
            caption = backstory_data.get('caption', '')
            
            logger.debug(f"Processing backstory {story_id}: {character} in {book_name}")
            
            evidence = self.extractor.extract_evidence(
                book_name=book_name,
                character=character,
                pipeline=self.pathway_pipeline
            )
            
            full_backstory = f"{caption}. {content}" if caption else content
            analysis_result = self.analyzer.analyze(
                backstory=full_backstory,
                book_name=book_name,
                character=character,
                evidence=evidence
            )
            
            consistency_result = self.checker.check_consistency(
                analysis_result=analysis_result,
                evidence=evidence
            )
            
            rationale = self.checker.generate_rationale(
                consistency_result=consistency_result,
                analysis_result=analysis_result
            )
            
            result = {
                'Story ID': story_id,
                'Prediction': consistency_result.get('prediction', 0),
                'Rationale': rationale,
                'Confidence': consistency_result.get('confidence', 0.5),
                'Book': book_name,
                'Character': character,
                'Evidence_Count': len(evidence.get('passages', [])),
                'Timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing backstory {backstory_data.get('id')}: {e}")
            return {
                'Story ID': backstory_data.get('id', 0),
                'Prediction': 0,
                'Rationale': f"Error during analysis: {str(e)[:100]}",
                'Confidence': 0.0,
                'Book': backstory_data.get('book_name', 'Unknown'),
                'Character': backstory_data.get('char', 'Unknown'),
                'Evidence_Count': 0,
                'Timestamp': datetime.now().isoformat()
            }
    
    def run_batch_analysis(self, test_data, train_data=None, output_path="output/results.csv"):
        logger.info(f"Starting batch analysis on {len(test_data)} backstories")
        
        if train_data is not None and len(train_data) > 0:
            logger.info(f"Learning from {len(train_data)} training examples")
            self.checker.learn_from_training(train_data)
        
        results = []
        for idx, backstory in enumerate(test_data):
            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{len(test_data)} backstories")
            
            result = self.process_backstory(backstory)
            results.append(result)
        
        logger.info(f"Completed analysis of {len(results)} backstories")
        
        save_results(results, output_path)
        
        return results

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="KDSH 2026 - Narrative Consistency Analysis System (Track A)"
    )
    
    parser.add_argument('--test', type=str, default='data/test.csv', help='Path to test CSV file')
    parser.add_argument('--train', type=str, default=None, help='Path to training CSV file')
    parser.add_argument('--output', type=str, default='output/results.csv', help='Output file path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--mode', type=str, choices=['fast', 'accurate'], default='fast', help='Processing mode')
    parser.add_argument('--report', action='store_true', help='Generate detailed analysis report')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--sample', type=int, default=None, help='Process only a sample of backstories')
    
    return parser.parse_args()

def main():
    print("\n" + "="*70)
    print("KDSH 2026 - NARRATIVE CONSISTENCY ANALYSIS SYSTEM")
    print("Track A: Pathway-based Solution")
    print("="*70 + "\n")
    
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        logger.info(f"Initializing system with config: {args.config}")
        system = NarrativeConsistencySystem(config_path=args.config)
        
        if not validate_inputs(args.test):
            logger.error(f"Invalid test file: {args.test}")
            sys.exit(1)
        
        test_data = pd.read_csv(args.test).to_dict('records')
        
        train_data = None
        if args.train and os.path.exists(args.train):
            train_data = pd.read_csv(args.train).to_dict('records')
        
        if args.sample and args.sample < len(test_data):
            logger.info(f"Processing sample of {args.sample} backstories")
            test_data = test_data[:args.sample]
        
        system.initialize_components(mode=args.mode)
        
        logger.info(f"Starting analysis in {args.mode} mode...")
        results = system.run_batch_analysis(
            test_data=test_data,
            train_data=train_data,
            output_path=args.output
        )
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*70)
        
        total = len(results)
        consistent = sum(1 for r in results if r['Prediction'] == 1)
        avg_confidence = sum(r['Confidence'] for r in results) / total if total > 0 else 0
        
        print(f"\nResults Summary:")
        print(f"  Total predictions: {total}")
        print(f"  Consistent (1): {consistent} ({consistent/total*100:.1f}%)")
        print(f"  Contradict (0): {total - consistent} ({(total-consistent)/total*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.2f}")
        
        print(f"\nOutput files:")
        print(f"  Results CSV: {args.output}")
        print(f"  Log file: analysis.log")
        
        print("\nSample predictions:")
        print("-"*70)
        for result in results[:5]:
            pred_text = "CONSISTENT" if result['Prediction'] == 1 else "CONTRADICT"
            print(f"ID {result['Story ID']}: {pred_text}")
            print(f"  Character: {result['Character']} in {result['Book']}")
            print(f"  Rationale: {result['Rationale']}")
            print(f"  Confidence: {result['Confidence']:.2f}")
            print()
        
        print("="*70)
        print("Analysis completed successfully!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())