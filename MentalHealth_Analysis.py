import re
from typing import Dict, List, Tuple, Optional
import json

class MentalHealthMonitor:
        
    def __init__(self):
        # Define mental health indicators with severity levels (0: neutral, 1: mild, 2: moderate, 3: severe)
        self.indicators = {
            # Depression indicators
            "depression": {
                "keywords": ["depressed", "hopeless", "worthless", "empty", "sad all the time", 
                            "no energy", "can't feel happy", "suicidal", "want to die", 
                            "life is pointless", "no reason to live"],
                "patterns": [
                    r"(feel|am|i'm)\s+(so|very|extremely)?\s*(depressed|hopeless|empty)",
                    r"(want to|thinking of|considering)\s+(suicide|ending it all|killing myself)",
                    r"(no|nothing)\s+(point|reason|meaning)\s+(to live|in life)",
                    r"can'?t\s+(get out of bed|find energy|feel anything)"
                ],
                "severity_weights": {
                    "suicidal": 3,
                    "want to die": 3,
                    "hopeless": 2,
                    "depressed": 2,
                    "no energy": 1,
                    "sad": 1
                }
            },
            
            # Anxiety indicators
            "anxiety": {
                "keywords": ["anxious", "panic", "worried", "overthinking", "can't stop thinking",
                            "nervous", "scared", "afraid", "heart racing", "panic attack",
                            "constant worry", "fear"],
                "patterns": [
                    r"(panic|anxiety)\s+attack",
                    r"(constant|always)\s+worr(?:y|ied)",
                    r"can'?t\s+(stop thinking|calm down|relax)",
                    r"(heart|chest)\s+(racing|pounding|tight)"
                ],
                "severity_weights": {
                    "panic attack": 3,
                    "constant worry": 2,
                    "anxious": 1,
                    "nervous": 1
                }
            },
            
            # Stress indicators
            "stress": {
                "keywords": ["stressed", "overwhelmed", "burnout", "burnt out", "too much pressure",
                            "can't handle", "exhausted", "drained", "too many responsibilities"],
                "patterns": [
                    r"(too much|overwhelming)\s+(pressure|work|responsibility)",
                    r"(burn(?:t)? out|exhausted|drained)",
                    r"can'?t\s+(handle|cope|deal with)",
                    r"(stressed|overwhelmed)\s+(to the point|so much)"
                ],
                "severity_weights": {
                    "burnout": 3,
                    "can't handle": 2,
                    "overwhelmed": 2,
                    "stressed": 1
                }
            },
            
            # Loneliness indicators
            "loneliness": {
                "keywords": ["lonely", "alone", "isolated", "no friends", "no one understands",
                            "feel alone", "empty", "isolated", "no connection"],
                "patterns": [
                    r"(feel|am)\s+(so|very)?\s*(alone|lonely|isolated)",
                    r"no\s+(one|friend)\s+(understand|care|talk to)",
                    r"(always|constantly)\s+by myself",
                    r"feel\s+like\s+i'?m\s+(the only one|completely alone)"
                ],
                "severity_weights": {
                    "completely alone": 3,
                    "no one cares": 2,
                    "lonely": 1,
                    "isolated": 1
                }
            },
            
            # Anger/Frustration indicators
            "anger": {
                "keywords": ["angry", "frustrated", "irritated", "mad", "pissed off", "rage",
                            "can't control anger", "losing temper", "short fuse"],
                "patterns": [
                    r"(so|very)\s+(angry|mad|pissed)",
                    r"can'?t\s+(control|manage)\s+(my|this)\s+anger",
                    r"(always|constantly)\s+(irritated|frustrated)",
                    r"(lose|losing)\s+my\s+temper"
                ],
                "severity_weights": {
                    "rage": 3,
                    "can't control anger": 3,
                    "losing temper": 2,
                    "angry": 1
                }
            }
        }
        
        # Crisis keywords that need immediate attention
        self.crisis_keywords = [
            "suicide", "kill myself", "end it all", "want to die", 
            "harm myself", "self harm", "can't go on", "ending my life"
        ]
        
        # Positive indicators (resilience, coping)
        self.positive_indicators = [
            "coping", "managing", "getting better", "improving",
            "feeling better", "positive", "hopeful", "strong",
            "resilient", "grateful", "thankful", "peaceful"
        ]
    
    def analyze_message(self, message: str) -> Dict:
        
        message_lower = message.lower().strip()
        
        results = {
            "conditions_detected": [],
            "severity_scores": {},
            "overall_severity": 0,
            "crisis_flag": False,
            "positive_signs": False,
            "detailed_analysis": {}
        }
        
        # Check for crisis keywords first (immediate attention needed)
        for keyword in self.crisis_keywords:
            if keyword in message_lower:
                results["crisis_flag"] = True
                results["overall_severity"] = 3  # Highest severity
                results["conditions_detected"].append("CRISIS")
                break
        
        # Analyze each condition
        for condition, data in self.indicators.items():
            severity_score = 0
            detected_keywords = []
            matched_patterns = []
            
            # Check keywords
            for keyword in data["keywords"]:
                if keyword in message_lower:
                    detected_keywords.append(keyword)
                    # Add severity weight
                    for weight_key, weight in data["severity_weights"].items():
                        if weight_key in keyword:
                            severity_score = max(severity_score, weight)
            
            # Check regex patterns
            for pattern in data["patterns"]:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    matched_patterns.append(pattern)
                    severity_score = max(severity_score, 2)  # Patterns indicate stronger signal
            
            # If condition detected
            if detected_keywords or matched_patterns:
                results["conditions_detected"].append(condition)
                results["severity_scores"][condition] = severity_score
                results["overall_severity"] = max(results["overall_severity"], severity_score)
                
                results["detailed_analysis"][condition] = {
                    "keywords_found": detected_keywords,
                    "patterns_matched": matched_patterns,
                    "severity": severity_score
                }
        
        # Check for positive indicators
        positive_count = 0
        for indicator in self.positive_indicators:
            if indicator in message_lower:
                positive_count += 1
        
        if positive_count >= 2:
            results["positive_signs"] = True
        
        return results
    
    def generate_alert(self, analysis: Dict, user_id: str = None) -> Optional[str]:
       
        alerts = []
        
        if analysis["crisis_flag"]:
            alerts.append("🚨 CRISIS ALERT: User expressing suicidal thoughts. Immediate professional help needed.")
        
        # Check for high severity conditions
        for condition, severity in analysis["severity_scores"].items():
            if severity >= 3:
                alerts.append(f"⚠️ HIGH SEVERITY: {condition.upper()} detected (severity: {severity})")
            elif severity == 2:
                alerts.append(f"⚠️ MODERATE: {condition.capitalize()} indicators present")
        
        # Add recommendations
        if alerts:
            alerts.append("\nRecommended actions:")
            
            if analysis["crisis_flag"]:
                alerts.append("- Provide crisis hotline numbers immediately")
                alerts.append("- Encourage immediate professional help")
                alerts.append("- Don't leave user alone if possible")
            elif analysis["overall_severity"] >= 2:
                alerts.append("- Suggest professional counseling")
                alerts.append("- Provide coping strategies")
                alerts.append("- Check in frequently")
            else:
                alerts.append("- Provide supportive guidance")
                alerts.append("- Suggest self-care practices")
        
        return "\n".join(alerts) if alerts else None
    
    def get_response_guidance(self, analysis: Dict) -> Dict:
        guidance = {
            "tone": "neutral",
            "priority_topics": [],
            "avoid_topics": [],
            "recommended_actions": [],
            "safety_notice": False
        }
        
        if analysis["crisis_flag"]:
            guidance["tone"] = "calm_urgent"
            guidance["priority_topics"] = ["safety", "professional_help", "immediate_support"]
            guidance["avoid_topics"] = ["philosophical", "abstract", "judgmental"]
            guidance["recommended_actions"] = ["provide_hotlines", "encourage_professional_help"]
            guidance["safety_notice"] = True
        
        elif analysis["overall_severity"] >= 2:
            guidance["tone"] = "compassionate_serious"
            guidance["priority_topics"] = ["validation", "coping_strategies", "professional_help"]
            guidance["recommended_actions"] = ["suggest_therapy", "provide_resources"]
        
        elif analysis["overall_severity"] == 1:
            guidance["tone"] = "supportive"
            guidance["priority_topics"] = ["validation", "self_care", "mindfulness"]
            guidance["recommended_actions"] = ["suggest_self_care", "encourage_social_support"]
        
        elif analysis["positive_signs"]:
            guidance["tone"] = "encouraging"
            guidance["priority_topics"] = ["reinforcement", "growth", "resilience"]
            guidance["recommended_actions"] = ["acknowledge_progress", "encourage_continuation"]
        
        return guidance