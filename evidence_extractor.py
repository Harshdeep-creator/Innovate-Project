import os
import re
from typing import Dict, List, Any

class EvidenceExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        
        self.narrative_summaries = {
            "The Count of Monte Cristo": {
                "summary": "Post-Napoleonic revenge tale of Edmond Dantès...",
                "key_characters": ["Edmond Dantès", "Abbé Faria", "Mercédès", "Noirtier"],
                "key_events": ["betrayal", "imprisonment", "escape", "revenge"]
            },
            "In Search of the Castaways": {
                "summary": "Global rescue mission for Captain Grant...",
                "key_characters": ["Jacques Paganel", "Lord Glenarvan", "Tom Ayrton", "Thalcave"],
                "key_events": ["bottle discovery", "expedition", "rescue"]
            }
        }
    
    def extract_evidence(self, book_name: str, character: str, pipeline: Any = None) -> Dict[str, Any]:
        evidence = {
            'book': book_name,
            'character': character,
            'passages': [],
            'metadata': {},
            'source': 'unknown'
        }
        
        try:
            if pipeline and hasattr(pipeline, 'query_character_passages'):
                passages = pipeline.query_character_passages(
                    book_name=book_name,
                    character_name=character,
                    max_results=10
                )
                evidence['passages'] = passages
                evidence['source'] = 'pathway'
                evidence['metadata']['pipeline_used'] = True
            
            if not evidence['passages']:
                evidence['passages'] = self._fallback_extraction(book_name, character)
                evidence['source'] = 'fallback'
            
            evidence['mention_count'] = self._count_character_mentions(
                evidence['passages'], character
            )
            
            evidence['timeline_hints'] = self._extract_timeline_hints(
                evidence['passages']
            )
            
        except Exception as e:
            evidence['error'] = str(e)
        
        return evidence
    
    def _fallback_extraction(self, book_name: str, character: str) -> List[Dict[str, Any]]:
        passages = []
        
        character_passages = {
            "The Count of Monte Cristo": {
                "Noirtier": [
                    "Noirtier was a political figure during the Napoleonic era.",
                    "Despite paralysis, Noirtier remained mentally sharp.",
                    "Noirtier had a complex relationship with his son Villefort."
                ],
                "Faria": [
                    "Abbé Faria was imprisoned in Château d'If for many years.",
                    "Faria educated Edmond Dantès in languages and sciences.",
                    "Faria revealed the Monte Cristo treasure location before dying."
                ]
            },
            "In Search of the Castaways": {
                "Paganel": [
                    "Jacques Paganel was an eccentric French geographer.",
                    "Paganel's geographic knowledge was crucial for the rescue.",
                    "He accidentally joined the expedition but proved invaluable."
                ],
                "Thalcave": [
                    "Thalcave was a skilled Patagonian guide.",
                    "He was an expert horseman and knew the terrain well.",
                    "Thalcave's bravery was instrumental to the mission."
                ]
            }
        }
        
        book_passages = character_passages.get(book_name, {})
        char_passages = book_passages.get(character, [])
        
        for i, text in enumerate(char_passages):
            passages.append({
                'text': text,
                'book': book_name,
                'character': character,
                'relevance_score': 0.8 - (i * 0.1),
                'source': 'fallback_knowledge_base',
                'position': i
            })
        
        if not passages:
            book_summary = self.narrative_summaries.get(book_name, {}).get('summary', '')
            if book_summary and character.lower() in book_summary.lower():
                passages.append({
                    'text': book_summary[:200] + '...',
                    'book': book_name,
                    'character': character,
                    'relevance_score': 0.5,
                    'source': 'book_summary',
                    'position': 0
                })
        
        return passages
    
    def _count_character_mentions(self, passages: List[Dict[str, Any]], character: str) -> int:
        if not passages:
            return 0
        
        count = 0
        for passage in passages:
            text = passage.get('text', '')
            count += text.lower().count(character.lower())
            
            words = character.split()
            for word in words:
                if len(word) > 3:
                    count += text.lower().count(word.lower())
        
        return count
    
    def _extract_timeline_hints(self, passages: List[Dict[str, Any]]) -> List[str]:
        timeline_hints = []
        
        time_patterns = [
            r'(?:early|mid|late)\s+\d+(?:st|nd|rd|th)?\s+century',
            r'\d{4}s?',
            r'(?:before|after|during)\s+[A-Z][a-z]+',
            r'(?:year|in)\s+\d{4}'
        ]
        
        for passage in passages:
            text = passage.get('text', '')
            for pattern in time_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                timeline_hints.extend(matches)
        
        return list(set(timeline_hints))[:5]
    
    def extract_from_file(self, file_path: str, character: str, max_passages: int = 20) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            paragraphs = text.split('\n\n')
            passages = []
            
            for i, para in enumerate(paragraphs):
                if character.lower() in para.lower():
                    mentions = para.lower().count(character.lower())
                    relevance = min(mentions * 0.2, 1.0)
                    
                    passages.append({
                        'text': para.strip(),
                        'position': i,
                        'relevance_score': relevance,
                        'source': 'full_text',
                        'paragraph_index': i
                    })
                    
                    if len(passages) >= max_passages:
                        break
            
            return passages
            
        except Exception as e:
            return []
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_word': i,
                'end_word': i + len(chunk_words),
                'word_count': len(chunk_words),
                'chunk_index': len(chunks)
            })
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks