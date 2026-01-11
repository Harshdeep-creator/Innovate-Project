import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ConsistencyResult:
    prediction: int
    confidence: float
    scores: Dict[str, float]
    decision_factors: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.decision_factors is None:
            self.decision_factors = []

class ConsistencyChecker:
    def __init__(self, config: Dict[str, Any], mode: str = "accurate"):
        self.config = config
        self.mode = mode
        self.weights = config.get('weights', {})
        self.thresholds = config.get('thresholds', {})
        
        self.character_patterns = defaultdict(lambda: {
            'consistent_keywords': [],
            'contradict_keywords': [],
            'consistent_examples': 0,
            'contradict_examples': 0
        })
        
        self.character_rules = config.get('character_rules', {})
    
    def learn_from_training(self, training_data: List[Dict[str, Any]]):
        if not training_data:
            return
        
        for example in training_data:
            book = example.get('book_name')
            character = example.get('char')
            content = str(example.get('content', ''))
            label = example.get('label', '')
            
            if not all([book, character, content, label]):
                continue
            
            key = f"{book}_{character}"
            
            words = content.lower().split()
            keywords = [word for word in words if len(word) > 3 and word.isalpha()]
            
            if label == 'consistent':
                self.character_patterns[key]['consistent_keywords'].extend(keywords[:5])
                self.character_patterns[key]['consistent_examples'] += 1
            elif label == 'contradict':
                self.character_patterns[key]['contradict_keywords'].extend(keywords[:5])
                self.character_patterns[key]['contradict_examples'] += 1
        
        for key in self.character_patterns:
            self.character_patterns[key]['consistent_keywords'] = list(set(
                self.character_patterns[key]['consistent_keywords']
            ))[:10]
            self.character_patterns[key]['contradict_keywords'] = list(set(
                self.character_patterns[key]['contradict_keywords']
            ))[:10]
    
    def check_consistency(self, analysis_result: Any, evidence: Dict[str, Any]) -> ConsistencyResult:
        try:
            scores = self._extract_scores(analysis_result)
            
            char_rules_score = self._apply_character_rules(analysis_result, evidence)
            scores['rules'] = char_rules_score
            
            pattern_score = self._check_learned_patterns(analysis_result, evidence)
            scores['patterns'] = pattern_score
            
            overall_score = self._calculate_overall_score(scores)
            
            prediction, confidence = self._make_prediction(overall_score, scores)
            
            decision_factors = self._identify_decision_factors(scores, overall_score)
            
            result = ConsistencyResult(
                prediction=prediction,
                confidence=confidence,
                scores=scores,
                decision_factors=decision_factors,
                metadata={
                    'evidence_count': len(evidence.get('passages', [])),
                    'mode': self.mode
                }
            )
            
            return result
            
        except Exception as e:
            return ConsistencyResult(
                prediction=0,
                confidence=0.0,
                scores={},
                decision_factors=[f"Error: {str(e)[:100]}"],
                metadata={'error': str(e)}
            )
    
    def _extract_scores(self, analysis_result: Any) -> Dict[str, float]:
        if hasattr(analysis_result, 'get_overall_score'):
            overall = analysis_result.get_overall_score(self.weights)
            scores = {
                'overall': overall,
                'confidence': getattr(analysis_result, 'confidence', 0.5)
            }
        elif hasattr(analysis_result, '__dict__'):
            scores = {
                'temporal': getattr(analysis_result, 'temporal_score', 0.5),
                'thematic': getattr(analysis_result, 'thematic_score', 0.5),
                'character': getattr(analysis_result, 'character_score', 0.5),
                'semantic': getattr(analysis_result, 'semantic_score', 0.5),
                'factual': getattr(analysis_result, 'factual_score', 0.5)
            }
            scores['overall'] = np.mean(list(scores.values()))
        else:
            scores = {'overall': 0.5}
        
        return scores
    
    def _apply_character_rules(self, analysis_result: Any, evidence: Dict[str, Any]) -> float:
        character = getattr(analysis_result, 'character', '')
        book = getattr(analysis_result, 'book', '')
        
        if not character or not book:
            return 0.5
        
        book_rules = self.character_rules.get(book, {})
        char_rules = book_rules.get(character, {})
        
        if not char_rules:
            return 0.5
        
        timeline = char_rules.get('timeline', '')
        constraints = char_rules.get('constraints', [])
        
        score = 0.7
        
        backstory = getattr(analysis_result, 'backstory', '')
        if timeline and timeline.lower() in backstory.lower():
            score += 0.2
        
        constraint_matches = 0
        for constraint in constraints:
            if constraint.lower() in backstory.lower():
                constraint_matches += 1
        
        if constraints:
            score += (constraint_matches / len(constraints)) * 0.3
        
        return min(score, 1.0)
    
    def _check_learned_patterns(self, analysis_result: Any, evidence: Dict[str, Any]) -> float:
        character = getattr(analysis_result, 'character', '')
        book = getattr(analysis_result, 'book', '')
        backstory = getattr(analysis_result, 'backstory', '')
        
        if not all([character, book, backstory]):
            return 0.5
        
        key = f"{book}_{character}"
        patterns = self.character_patterns.get(key)
        
        if not patterns:
            return 0.5
        
        backstory_lower = backstory.lower()
        
        consistent_matches = 0
        contradict_matches = 0
        
        for keyword in patterns.get('consistent_keywords', []):
            if keyword in backstory_lower:
                consistent_matches += 1
        
        for keyword in patterns.get('contradict_keywords', []):
            if keyword in backstory_lower:
                contradict_matches += 1
        
        total_matches = consistent_matches + contradict_matches
        if total_matches == 0:
            return 0.5
        
        pattern_score = consistent_matches / total_matches
        
        total_examples = patterns['consistent_examples'] + patterns['contradict_examples']
        if total_examples > 0:
            weight = min(total_examples / 10, 1.0)
            pattern_score = (pattern_score * weight) + (0.5 * (1 - weight))
        
        return pattern_score
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        if 'overall' in scores:
            return scores['overall']
        
        component_weights = {
            'temporal': self.weights.get('temporal', 0.2),
            'thematic': self.weights.get('thematic', 0.2),
            'character': self.weights.get('character', 0.2),
            'semantic': self.weights.get('semantic', 0.15),
            'factual': self.weights.get('factual', 0.15),
            'rules': 0.05,
            'patterns': 0.05
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for component, weight in component_weights.items():
            if component in scores:
                weighted_sum += scores[component] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        overall_score = weighted_sum / total_weight
        return overall_score
    
    def _make_prediction(self, overall_score: float, scores: Dict[str, float]) -> Tuple[int, float]:
        threshold = self.thresholds.get('consistency', 0.6)
        
        prediction = 1 if overall_score >= threshold else 0
        
        confidence = abs(overall_score - 0.5) * 2
        
        if 'confidence' in scores:
            confidence = (confidence + scores['confidence']) / 2
        
        return prediction, min(confidence, 1.0)
    
    def _identify_decision_factors(self, scores: Dict[str, float], overall_score: float) -> List[str]:
        factors = []
        
        for component in ['temporal', 'thematic', 'character', 'factual']:
            if component in scores:
                score = scores[component]
                if score > 0.7:
                    factors.append(f"Strong {component} alignment")
                elif score < 0.3:
                    factors.append(f"Weak {component} alignment")
        
        if overall_score > 0.7:
            factors.append("High overall consistency")
        elif overall_score < 0.4:
            factors.append("Low overall consistency")
        
        return factors[:3]
    
    def generate_rationale(self, consistency_result: ConsistencyResult, analysis_result: Any) -> str:
        prediction = consistency_result.prediction
        confidence = consistency_result.confidence
        factors = consistency_result.decision_factors
        
        if prediction == 1:
            base = "Consistent: "
            if confidence > 0.8:
                base += "Strong evidence supports backstory"
            elif confidence > 0.6:
                base += "Moderate evidence supports backstory"
            else:
                base += "Limited evidence but plausible"
        else:
            base = "Contradicts: "
            if confidence > 0.8:
                base += "Clear contradictions found"
            elif confidence > 0.6:
                base += "Multiple inconsistencies"
            else:
                base += "Weak alignment with narrative"
        
        if factors:
            base += f" ({', '.join(factors)})"
        
        if len(base) > 200:
            base = base[:197] + "..."
        
        return base