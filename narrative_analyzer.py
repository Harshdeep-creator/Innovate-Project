import re
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class AnalysisResult:
    temporal_score: float = 0.0
    thematic_score: float = 0.0
    character_score: float = 0.0
    semantic_score: float = 0.0
    factual_score: float = 0.0
    confidence: float = 0.0
    key_findings: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.key_findings is None:
            self.key_findings = []
        if self.warnings is None:
            self.warnings = []
    
    def get_overall_score(self, weights: Dict[str, float]) -> float:
        scores = {
            'temporal': self.temporal_score,
            'thematic': self.thematic_score,
            'character': self.character_score,
            'semantic': self.semantic_score,
            'factual': self.factual_score
        }
        
        total = sum(scores[key] * weights.get(key, 0.2) for key in scores)
        return total / sum(weights.values()) if weights else total

class NarrativeAnalyzer:
    def __init__(self, config: Dict[str, Any], narrative_knowledge: Dict[str, Any]):
        self.config = config
        self.narrative_knowledge = narrative_knowledge
        
        self.embedding_model = None
        self._init_models()
        
        self.character_profiles = defaultdict(dict)
    
    def _init_models(self):
        try:
            model_name = self.config.get('embedding', {}).get('name', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def analyze(self, backstory: str, book_name: str, character: str, evidence: Dict[str, Any]) -> AnalysisResult:
        result = AnalysisResult()
        
        try:
            result.temporal_score = self._analyze_temporal(backstory, book_name, character)
            result.thematic_score = self._analyze_thematic(backstory, book_name)
            result.character_score = self._analyze_character(backstory, book_name, character, evidence)
            result.semantic_score = self._analyze_semantic(backstory, evidence)
            result.factual_score = self._analyze_factual(backstory, book_name, character, evidence)
            
            result.confidence = self._calculate_confidence(result)
            result.key_findings = self._generate_findings(result, backstory, character)
            
        except Exception as e:
            result.warnings.append(f"Analysis error: {str(e)[:100]}")
        
        return result
    
    def _analyze_temporal(self, backstory: str, book_name: str, character: str) -> float:
        book_info = self.narrative_knowledge.get(book_name, {})
        book_period = book_info.get('period', '')
        
        time_patterns = {
            'century': r'(\d+)(?:st|nd|rd|th)? century',
            'year': r'(?:year|in) (\d{4})',
            'era': r'(?:medieval|renaissance|enlightenment|victorian|modern)',
            'historical': r'(?:napoleon|world war|revolution)'
        }
        
        time_matches = []
        for pattern_name, pattern in time_patterns.items():
            matches = re.findall(pattern, backstory.lower())
            if matches:
                time_matches.extend(matches)
        
        if not time_matches:
            return 0.7
        
        if 'monte cristo' in book_name.lower():
            if any('19' in str(match) or '180' in str(match) for match in time_matches):
                return 0.9
            if any('20' in str(match) or 'world war' in str(match) for match in time_matches):
                return 0.1
        
        elif 'castaways' in book_name.lower():
            if any('19' in str(match) for match in time_matches):
                return 0.9
            if any('20' in str(match) for match in time_matches):
                return 0.2
        
        return 0.5
    
    def _analyze_thematic(self, backstory: str, book_name: str) -> float:
        book_info = self.narrative_knowledge.get(book_name, {})
        book_themes = book_info.get('themes', [])
        
        if not book_themes:
            return 0.5
        
        backstory_lower = backstory.lower()
        
        theme_matches = 0
        for theme in book_themes:
            if theme in backstory_lower:
                theme_matches += 1
        
        score = theme_matches / len(book_themes)
        return min(score * 1.5, 1.0)
    
    def _analyze_character(self, backstory: str, book_name: str, character: str, evidence: Dict[str, Any]) -> float:
        character_profile = self._build_character_profile(character, evidence)
        
        profile_text = ' '.join(character_profile.values())
        
        if not profile_text:
            return 0.5
        
        if self.embedding_model:
            embeddings = self.embedding_model.encode([backstory, profile_text])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity
        
        return 0.5
    
    def _build_character_profile(self, character: str, evidence: Dict[str, Any]) -> Dict[str, str]:
        profile = {}
        
        passages = evidence.get('passages', [])
        character_texts = []
        
        for passage in passages:
            if isinstance(passage, dict) and 'text' in passage:
                character_texts.append(passage['text'])
        
        if character_texts:
            profile['evidence'] = ' '.join(character_texts[:3])
        
        return profile
    
    def _analyze_semantic(self, backstory: str, evidence: Dict[str, Any]) -> float:
        if not self.embedding_model:
            return 0.5
        
        evidence_texts = []
        for passage in evidence.get('passages', []):
            if isinstance(passage, dict) and 'text' in passage:
                evidence_texts.append(passage['text'])
        
        if not evidence_texts:
            return 0.5
        
        backstory_embedding = self.embedding_model.encode([backstory])[0]
        evidence_embeddings = self.embedding_model.encode(evidence_texts)
        
        similarities = cosine_similarity([backstory_embedding], evidence_embeddings)[0]
        return float(np.mean(similarities))
    
    def _analyze_factual(self, backstory: str, book_name: str, character: str, evidence: Dict[str, Any]) -> float:
        contradictions = self._detect_contradictions(backstory, book_name, character)
        
        if contradictions:
            score = 1.0 / (1 + len(contradictions))
            return score
        
        return 0.7
    
    def _detect_contradictions(self, backstory: str, book_name: str, character: str) -> List[str]:
        contradictions = []
        backstory_lower = backstory.lower()
        
        contradiction_rules = {
            "The Count of Monte Cristo": {
                "Faria": ["escaped prison", "was free", "lived freely"],
                "Noirtier": ["was healthy", "could speak", "wasn't paralyzed"]
            },
            "In Search of the Castaways": {
                "Paganel": ["was British", "hated geography", "stayed in Europe"],
                "Ayrton": ["was honest", "was loyal", "helped Grant willingly"]
            }
        }
        
        book_rules = contradiction_rules.get(book_name, {})
        char_rules = book_rules.get(character, [])
        
        for rule in char_rules:
            if rule in backstory_lower:
                contradictions.append(rule)
        
        return contradictions
    
    def _calculate_confidence(self, result: AnalysisResult) -> float:
        scores = [
            result.temporal_score,
            result.thematic_score,
            result.character_score,
            result.semantic_score,
            result.factual_score
        ]
        
        weights = [0.2, 0.2, 0.25, 0.2, 0.15]
        
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        return weighted_sum
    
    def _generate_findings(self, result: AnalysisResult, backstory: str, character: str) -> List[str]:
        findings = []
        
        if result.temporal_score > 0.8:
            findings.append(f"Strong temporal alignment for {character}")
        elif result.temporal_score < 0.3:
            findings.append(f"Potential temporal issues for {character}")
        
        if result.thematic_score > 0.7:
            findings.append("Good thematic alignment with narrative")
        
        if result.character_score > 0.6:
            findings.append(f"Character consistency confirmed")
        
        return findings