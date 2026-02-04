import torch
from typing import Dict, List, Optional, Tuple
import json
import re
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class TherapeuticStrategy(Enum):
    ASKING_QUESTIONS = "asking_questions"
    REFLECTIVE_LISTENING = "reflective_listening"
    GIVING_ADVICE = "giving_advice"
    ASSESSMENT = "assessment"  # Added for monitoring
    CRISIS_INTERVENTION = "crisis_intervention"  # Added for flags

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"

@dataclass
class MentalHealthAssessment:
    risk_level: RiskLevel
    detected_conditions: List[str]
    severity_score: float  # 0-3 scale from paper
    crisis_indicators: List[str]
    positive_signs: List[str]
    timestamp: datetime
    conversation_context: str
    recommended_action: str

class SSAGMentalHealthMonitor:
    """
    Implementation of SSAG (Script-Strategy Aligned Generation) for mental health monitoring
    Based on: Script-Strategy Aligned Generation: Aligning LLMs with Expert-Crafted Dialogue Scripts
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        self.user_profile = {
            "mental_health_history": [],
            "risk_trajectory": [],
            "intervention_history": [],
            "baseline_assessment": None
        }
        
        # Therapeutic topics from paper (Table 8 in Appendix A.1)
        self.therapeutic_topics = {
            "depression": {
                "description": "Addressing persistent low mood, hopelessness, and lack of energy",
                "questions": [
                    "How have you been feeling about yourself lately?",
                    "What activities have you been able to enjoy recently?",
                    "How has your energy level been?",
                    "Have you noticed changes in your sleep or appetite?"
                ],
                "advice": [
                    "It might help to establish a gentle daily routine",
                    "Consider reaching out to someone you trust about how you're feeling",
                    "Small, achievable goals can help build momentum"
                ]
            },
            "anxiety": {
                "description": "Managing excessive worry, nervousness, and physical symptoms",
                "questions": [
                    "What situations have been causing you stress or worry?",
                    "How does your body feel when you're anxious?",
                    "What helps you feel calmer in stressful moments?",
                    "Have you noticed patterns in what triggers your anxiety?"
                ],
                "advice": [
                    "Deep breathing exercises can help calm the nervous system",
                    "Mindfulness practices can help you stay grounded in the present",
                    "Breaking overwhelming tasks into smaller steps can reduce anxiety"
                ]
            },
            "stress": {
                "description": "Coping with overwhelming demands and pressure",
                "questions": [
                    "What aspects of your life feel most overwhelming right now?",
                    "How are you currently managing your stress?",
                    "What support systems do you have in place?",
                    "What would make your load feel more manageable?"
                ],
                "advice": [
                    "Prioritizing tasks and setting boundaries can help manage stress",
                    "Regular breaks and self-care are important during stressful periods",
                    "Sometimes delegating tasks can lighten your load"
                ]
            },
            "crisis": {
                "description": "Immediate risk assessment and intervention",
                "questions": [
                    "Are you having thoughts of harming yourself or others?",
                    "Do you feel safe right now?",
                    "Is there someone you can reach out to immediately?",
                    "What would help you feel more supported in this moment?"
                ],
                "advice": [
                    "Please contact emergency services or a crisis hotline immediately",
                    "Reach out to someone you trust and stay connected",
                    "Go to the nearest emergency room if you're in immediate danger"
                ],
                "emergency_resources": [
                    "National Suicide Prevention Lifeline: 988",
                    "Crisis Text Line: Text HOME to 741741",
                    "Emergency Services: 911"
                ]
            }
        }
        
        # Crisis indicators from paper (Section 5.1)
        self.crisis_indicators = {
            "suicidal_ideation": [
                "want to die", "kill myself", "end it all", "suicide",
                "no reason to live", "better off dead", "ending my life"
            ],
            "self_harm": [
                "harm myself", "hurt myself", "cutting", "self-harm"
            ],
            "hopelessness": [
                "nothing matters", "pointless", "no future", "no hope",
                "never get better", "stuck forever"
            ],
            "isolation": [
                "completely alone", "no one cares", "abandoned",
                "isolated from everyone", "no support"
            ]
        }
        
        # Positive indicators for resilience
        self.positive_indicators = {
            "coping": ["managing", "coping", "handling", "getting through"],
            "hope": ["hopeful", "optimistic", "better future", "things will improve"],
            "support": ["supportive", "helpful friends", "family helps", "have support"],
            "self_care": ["taking care", "self-care", "resting", "setting boundaries"]
        }
        
        # Monitoring flags from Study 2 results
        self.monitoring_thresholds = {
            "low_risk": {"severity": 0.0, "frequency": 0},
            "moderate_risk": {"severity": 1.0, "frequency": 3},  # 3 mentions in conversation
            "high_risk": {"severity": 2.0, "frequency": 5},
            "crisis": {"severity": 3.0, "frequency": 1}  # Immediate for crisis
        }

    def predict_therapeutic_strategy(self, conversation_context: str) -> List[TherapeuticStrategy]:
        """
        Step 1 of SSAG: Predict next therapeutic strategy based on context
        Based on Table 6 in paper: Therapeutic strategy prediction
        """
        # Prepare prompt based on Appendix B.3
        prompt = f"""Conversation context:
{conversation_context}

Therapeutic strategies:
'reflection': Mirroring client's feelings to promote self-reflection
'question': Gathering information or prompting self-exploration
'advice': Providing guidance or suggestions
'assessment': Evaluating mental state and risk factors
'crisis_intervention': Immediate safety-focused response

Task: Predict the next appropriate therapeutic strategy from the list above based on the conversation context.
Consider: current risk level, conversation flow, and therapeutic goals.

The next therapeutic strategy is:"""
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,  # Lower temperature for more predictable strategy selection
            do_sample=False
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response to extract strategies
        strategies = []
        for strategy in TherapeuticStrategy:
            if strategy.value in response.lower():
                strategies.append(strategy)
        
        # Default to assessment if none detected
        if not strategies:
            strategies.append(TherapeuticStrategy.ASSESSMENT)
            
        return strategies

    def analyze_message_for_risk(self, message: str) -> Dict:
        """
        Analyze individual message for mental health risk indicators
        Based on Study 2 methodology
        """
        message_lower = message.lower()
        
        analysis = {
            "crisis_flags": [],
            "risk_indicators": [],
            "positive_signs": [],
            "severity_score": 0.0,
            "requires_immediate_action": False
        }
        
        # Check for crisis indicators
        for category, indicators in self.crisis_indicators.items():
            for indicator in indicators:
                if indicator in message_lower:
                    analysis["crisis_flags"].append(category)
                    analysis["severity_score"] = max(analysis["severity_score"], 3.0)
                    analysis["requires_immediate_action"] = True
        
        # Check therapeutic topics for risk indicators
        for topic, data in self.therapeutic_topics.items():
            if topic == "crisis":
                continue
                
            # Check if message relates to topic
            topic_keywords = [
                topic,
                *[q.lower().split()[:3] for q in data["questions"]],
                *[a.lower().split()[:3] for a in data["advice"]]
            ]
            
            for keyword in topic_keywords:
                if isinstance(keyword, list):
                    if all(k in message_lower for k in keyword[:2]):
                        analysis["risk_indicators"].append(topic)
                        analysis["severity_score"] = max(analysis["severity_score"], 1.0)
                elif keyword in message_lower:
                    analysis["risk_indicators"].append(topic)
                    analysis["severity_score"] = max(analysis["severity_score"], 1.0)
        
        # Check for positive indicators
        for category, indicators in self.positive_indicators.items():
            for indicator in indicators:
                if indicator in message_lower:
                    analysis["positive_signs"].append(category)
                    # Positive signs reduce effective severity
                    analysis["severity_score"] = max(0, analysis["severity_score"] - 0.5)
        
        return analysis

    def assess_conversation_pattern(self, conversation: List[Dict]) -> MentalHealthAssessment:
        """ Assess entire conversation pattern for mental health risks
        Based on longitudinal tracking from Study 2
        """
        risk_scores = []
        all_crisis_flags = []
        all_risk_indicators = []
        all_positive_signs = []
        
        for turn in conversation:
            if turn["role"] == "user":
                analysis = self.analyze_message_for_risk(turn["content"])
                risk_scores.append(analysis["severity_score"])
                all_crisis_flags.extend(analysis["crisis_flags"])
                all_risk_indicators.extend(analysis["risk_indicators"])
                all_positive_signs.extend(analysis["positive_signs"])
        
        # Calculate overall risk based on paper thresholds
        avg_severity = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Determine risk level
        if any(flag for flag in all_crisis_flags):
            risk_level = RiskLevel.CRISIS
        elif avg_severity >= 2.0 or len(set(all_risk_indicators)) >= 3:
            risk_level = RiskLevel.HIGH
        elif avg_severity >= 1.0 or len(set(all_risk_indicators)) >= 2:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        # Generate recommended action based on risk level
        recommended_action = self._generate_recommendation(
            risk_level, all_crisis_flags, all_risk_indicators
        )
        
        assessment = MentalHealthAssessment(
            risk_level=risk_level,
            detected_conditions=list(set(all_risk_indicators)),
            severity_score=avg_severity,
            crisis_indicators=list(set(all_crisis_flags)),
            positive_signs=list(set(all_positive_signs)),
            timestamp=datetime.now(),
            conversation_context=self._summarize_conversation(conversation),
            recommended_action=recommended_action
        )
        
        # Update user profile
        self.user_profile["mental_health_history"].append(assessment)
        self.user_profile["risk_trajectory"].append({
            "timestamp": datetime.now(),
            "risk_level": risk_level.value,
            "severity": avg_severity
        })
        
        return assessment

    def _generate_recommendation(self, risk_level: RiskLevel, 
                                crisis_flags: List[str], 
                                risk_indicators: List[str]) -> str:
        
        recommendations = []
        
        if risk_level == RiskLevel.CRISIS:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED:")
            recommendations.append("- Contact emergency services or crisis hotline immediately")
            recommendations.append("- Do not leave the person alone if possible")
            recommendations.append("- Remove access to means of self-harm if applicable")
            recommendations.append("\nCrisis Resources:")
            recommendations.append("- National Suicide Prevention Lifeline: 988")
            recommendations.append("- Crisis Text Line: Text HOME to 741741")
            recommendations.append("- Go to nearest emergency room")
            
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("⚠️ URGENT PROFESSIONAL REFERRAL NEEDED:")
            recommendations.append("- Schedule appointment with mental health professional within 24-48 hours")
            recommendations.append("- Consider contacting primary care physician")
            recommendations.append("- Engage support system for monitoring")
            recommendations.append("\nMonitor for:")
            for indicator in risk_indicators:
                recommendations.append(f"- {indicator.capitalize()} symptoms")
                
        elif risk_level == RiskLevel.MODERATE:
            recommendations.append("📋 RECOMMEND PROFESSIONAL EVALUATION:")
            recommendations.append("- Consider consultation with mental health professional")
            recommendations.append("- Regular check-ins (every 2-3 days)")
            recommendations.append("- Implement coping strategies discussed")
            recommendations.append("\nSelf-care strategies:")
            recommendations.append("- Maintain daily routine")
            recommendations.append("- Practice mindfulness or relaxation techniques")
            recommendations.append("- Stay connected with supportive relationships")
            
        else:  # LOW risk
            recommendations.append("✅ CONTINUE WITH SUPPORTIVE MONITORING:")
            recommendations.append("- Regular self-check-ins")
            recommendations.append("- Maintain healthy lifestyle habits")
            recommendations.append("- Continue with current coping strategies")
            recommendations.append("- Reach out if symptoms worsen")
        
        return "\n".join(recommendations)

    def generate_monitored_response(self, system_message: str, 
                                   user_message: str) -> Dict:
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Analyze current message
        message_analysis = self.analyze_message_for_risk(user_message)
        
        # Assess overall conversation pattern
        conversation_assessment = self.assess_conversation_pattern(
            self.conversation_history[-10:]  # Last 10 messages
        )
        
        # Predict therapeutic strategy based on SSAG
        context_str = self._format_conversation_context(self.conversation_history[-5:])
        strategies = self.predict_therapeutic_strategy(context_str)
        
        # Override strategy if crisis detected
        if message_analysis["requires_immediate_action"]:
            strategies = [TherapeuticStrategy.CRISIS_INTERVENTION]
        
        # Generate enhanced system prompt with monitoring context
        enhanced_system = self._create_monitoring_prompt(
            base_system=system_message,
            assessment=conversation_assessment,
            strategies=strategies,
            message_analysis=message_analysis
        )
        
        # Generate response
        prompt = self._format_ssag_prompt(
            system_message=enhanced_system,
            history=self.conversation_history[-4:],
            strategies=strategies,
            assessment=conversation_assessment
        )
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Add crisis resources if needed
        if conversation_assessment.risk_level == RiskLevel.CRISIS:
            response += "\n\n" + self._get_crisis_resources()
        
        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Log intervention
        self.user_profile["intervention_history"].append({
            "timestamp": datetime.now(),
            "user_message": user_message,
            "assessment": conversation_assessment.__dict__,
            "response": response[:200],  # Truncate for storage
            "strategies_used": [s.value for s in strategies]
        })
        
        return {
            "response": response,
            "assessment": conversation_assessment,
            "message_analysis": message_analysis,
            "strategies_used": strategies,
            "flags_raised": self._generate_flags(conversation_assessment),
            "next_steps": self._determine_next_steps(conversation_assessment)
        }

    def _create_monitoring_prompt(self, base_system: str, 
                                 assessment: MentalHealthAssessment,
                                 strategies: List[TherapeuticStrategy],
                                 message_analysis: Dict) -> str:
        """
        Create enhanced prompt with monitoring context
        """
        enhanced = f"""{base_system}

MENTAL HEALTH MONITORING CONTEXT (Based on SSAG Methodology):
Current Risk Level: {assessment.risk_level.value.upper()}
Detected Conditions: {', '.join(assessment.detected_conditions) if assessment.detected_conditions else 'None detected'}
Severity Score: {assessment.severity_score:.1f}/3.0
Crisis Flags: {', '.join(assessment.crisis_indicators) if assessment.crisis_indicators else 'None'}
Positive Signs: {', '.join(assessment.positive_signs) if assessment.positive_signs else 'None'}

Selected Therapeutic Strategies: {', '.join([s.value for s in strategies])}

RESPONSE GUIDELINES:
{self._get_strategy_guidelines(strategies, assessment.risk_level)}

{"🚨 CRISIS PROTOCOL ACTIVATED: Prioritize safety and provide immediate resources" 
 if assessment.risk_level == RiskLevel.CRISIS else ""}

RECOMMENDED ACTION: {assessment.recommended_action.split('\n')[0]}
"""
        return enhanced

    def _get_strategy_guidelines(self, strategies: List[TherapeuticStrategy], 
                                risk_level: RiskLevel) -> str:
        """
        Get strategy-specific guidelines based on SSAG paper
        """
        guidelines = []
        
        for strategy in strategies:
            if strategy == TherapeuticStrategy.CRISIS_INTERVENTION:
                guidelines.append("1. Ensure immediate safety above all else")
                guidelines.append("2. Provide clear crisis resources")
                guidelines.append("3. Use calm, directive language")
                guidelines.append("4. Encourage immediate professional contact")
                guidelines.append("5. Avoid philosophical discussions")
            elif strategy == TherapeuticStrategy.ASSESSMENT:
                guidelines.append("1. Gently explore current mental state")
                guidelines.append("2. Assess risk factors without leading")
                guidelines.append("3. Validate feelings while gathering information")
                guidelines.append("4. Note protective factors and strengths")
                guidelines.append("5. Maintain therapeutic alliance")
            elif strategy == TherapeuticStrategy.REFLECTIVE_LISTENING:
                guidelines.append("1. Mirror feelings accurately")
                guidelines.append("2. Validate emotional experience")
                guidelines.append("3. Encourage self-exploration")
                guidelines.append("4. Build rapport through understanding")
                guidelines.append("5. Avoid interpretation or diagnosis")
            elif strategy == TherapeuticStrategy.ASKING_QUESTIONS:
                if risk_level in [RiskLevel.HIGH, RiskLevel.CRISIS]:
                    guidelines.append("1. Focus on safety assessment questions")
                    guidelines.append("2. Use open-ended but focused questions")
                    guidelines.append("3. Avoid overwhelming with too many questions")
                    guidelines.append("4. Prioritize questions about support systems")
                    guidelines.append("5. Check understanding and comfort level")
                else:
                    guidelines.append("1. Use therapeutic questions from script")
                    guidelines.append("2. Encourage self-reflection")
                    guidelines.append("3. Guide toward insight")
                    guidelines.append("4. Maintain conversational flow")
                    guidelines.append("5. Balance questions with reflections")
            elif strategy == TherapeuticStrategy.GIVING_ADVICE:
                if risk_level == RiskLevel.CRISIS:
                    guidelines.append("1. Limit advice to immediate safety measures")
                    guidelines.append("2. Focus on actionable crisis steps")
                    guidelines.append("3. Provide clear, simple instructions")
                    guidelines.append("4. Repeat critical information")
                    guidelines.append("5. Connect with professional help")
                else:
                    guidelines.append("1. Base advice on expert-crafted scripts")
                    guidelines.append("2. Tailor to individual context")
                    guidelines.append("3. Offer options, not directives")
                    guidelines.append("4. Empower self-directed change")
                    guidelines.append("5. Check receptiveness to suggestions")
        
        return "\n".join(guidelines)

    def _generate_flags(self, assessment: MentalHealthAssessment) -> List[Dict]:
        flags = []
        
        # Crisis flags
        if assessment.risk_level == RiskLevel.CRISIS:
            flags.append({
                "level": "CRITICAL",
                "type": "IMMEDIATE_CRISIS",
                "message": f"User shows crisis indicators: {', '.join(assessment.crisis_indicators)}",
                "action": "Requires immediate professional intervention",
                "timestamp": datetime.now().isoformat()
            })
        
        # High risk flags
        if assessment.risk_level == RiskLevel.HIGH:
            flags.append({
                "level": "HIGH",
                "type": "ELEVATED_RISK",
                "message": f"Multiple risk indicators detected: {', '.join(assessment.detected_conditions)}",
                "action": "Urgent referral to mental health professional needed",
                "timestamp": datetime.now().isoformat()
            })
        
        # Escalation flags (if risk is increasing)
        if len(self.user_profile["risk_trajectory"]) >= 2:
            recent_risks = self.user_profile["risk_trajectory"][-2:]
            if (recent_risks[1]["risk_level"] in ["high", "crisis"] and 
                recent_risks[0]["risk_level"] in ["low", "moderate"]):
                flags.append({
                    "level": "MEDIUM",
                    "type": "RISK_ESCALATION",
                    "message": "Risk level has escalated recently",
                    "action": "Increase monitoring frequency and consider check-in",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Pattern flags (recurring issues)
        if len(assessment.detected_conditions) >= 2:
            flags.append({
                "level": "MEDIUM",
                "type": "MULTIPLE_CONCERNS",
                "message": f"Multiple mental health concerns identified",
                "action": "Comprehensive assessment recommended",
                "timestamp": datetime.now().isoformat()
            })
        
        return flags

    def _determine_next_steps(self, assessment: MentalHealthAssessment) -> Dict:
        """
        Determine next steps based on assessment
        Based on Study 2 follow-up protocols
        """
        if assessment.risk_level == RiskLevel.CRISIS:
            return {
                "immediate": ["Contact emergency services", "Ensure safety"],
                "short_term": ["Crisis assessment", "Safety planning"],
                "follow_up": "Within 24 hours",
                "monitoring_frequency": "Continuous until stable"
            }
        elif assessment.risk_level == RiskLevel.HIGH:
            return {
                "immediate": ["Professional referral", "Support system activation"],
                "short_term": ["Comprehensive assessment", "Treatment planning"],
                "follow_up": "Within 48 hours",
                "monitoring_frequency": "Daily check-ins"
            }
        elif assessment.risk_level == RiskLevel.MODERATE:
            return {
                "immediate": ["Self-care plan", "Coping strategies"],
                "short_term": ["Professional consultation", "Regular monitoring"],
                "follow_up": "Within 3-5 days",
                "monitoring_frequency": "Every 2-3 days"
            }
        else:  # LOW
            return {
                "immediate": ["Maintain healthy habits", "Self-monitoring"],
                "short_term": ["Continue current strategies", "Regular check-ins"],
                "follow_up": "As needed",
                "monitoring_frequency": "Weekly check-ins"
            }

    def _format_conversation_context(self, history: List[Dict]) -> str:
        lines = []
        for msg in history:
            lines.append(f"{msg['role'].capitalize()}: {msg['content'][:100]}")
        return "\n".join(lines[-5:])  # Last 5 messages

    def _summarize_conversation(self, conversation: List[Dict]) -> str:
        user_messages = [msg["content"] for msg in conversation if msg["role"] == "user"]
        return " | ".join([msg[:50] for msg in user_messages[-3:]])  # Last 3 user messages

    def _format_ssag_prompt(self, system_message: str, history: List[Dict], 
                           strategies: List[TherapeuticStrategy],
                           assessment: MentalHealthAssessment) -> str:
        """Format prompt following SSAG methodology"""
        prompt = f"<|system|>\n{system_message}</s>\n"
        
        for msg in history:
            prompt += f"<|{msg['role']}|>\n{msg['content']}</s>\n"
        
        # Add strategy guidance
        strategy_text = " and ".join([s.value.replace('_', ' ') for s in strategies])
        prompt += f"<|strategy|>\nCurrent therapeutic strategy: {strategy_text}\n"
        prompt += f"Risk level: {assessment.risk_level.value}</s>\n"
        
        prompt += "<|assistant|>\n"
        
        return prompt

    def _get_crisis_resources(self) -> str:
        return """

        **CRISIS SUPPORT RESOURCES:**
        **Immediate Help:**
        - **National Suicide Prevention Lifeline**: {}
        - **Crisis Text Line**: Text {} to {}
        - **Emergency Services**: 

        **What to do right now:**
        1. Contact one of the resources above IMMEDIATELY
        2. Reach out to someone you trust
        3. Go to the nearest emergency room if you're in danger
        4. Remember: You don't have to go through this alone

        Your safety is the most important thing right now."""

    def generate_monitoring_report(self) -> Dict:
        if not self.user_profile["mental_health_history"]:
            return {"status": "No assessment data available"}
        
        latest_assessment = self.user_profile["mental_health_history"][-1]
        
        report = {
            "user_id": "anonymous",  # Would be actual user ID in production
            "generated_at": datetime.now().isoformat(),
            "assessment_period": {
                "start": self.user_profile["mental_health_history"][0].timestamp.isoformat(),
                "end": latest_assessment.timestamp.isoformat(),
                "duration_days": (latest_assessment.timestamp - 
                                 self.user_profile["mental_health_history"][0].timestamp).days
            },
            "current_status": {
                "risk_level": latest_assessment.risk_level.value,
                "severity_score": latest_assessment.severity_score,
                "active_concerns": latest_assessment.detected_conditions,
                "crisis_indicators": latest_assessment.crisis_indicators,
                "protective_factors": latest_assessment.positive_signs
            },
            "risk_trajectory": self.user_profile["risk_trajectory"],
            "intervention_summary": {
                "total_sessions": len(self.user_profile["intervention_history"]),
                "crisis_interventions": sum(1 for h in self.user_profile["intervention_history"] 
                                          if "CRISIS" in str(h.get("assessment", {}))),
                "common_strategies": self._get_common_strategies(),
                "response_patterns": self._analyze_response_patterns()
            },
            "recommendations": {
                "immediate": latest_assessment.recommended_action.split('\n')[:3],
                "follow_up": self._determine_next_steps(latest_assessment),
                "monitoring_plan": self._create_monitoring_plan(latest_assessment)
            },
            "flags_generated": self._generate_flags(latest_assessment)
        }
        
        return report

    def _get_common_strategies(self) -> List[str]:
        """Get most commonly used therapeutic strategies"""
        from collections import Counter
        all_strategies = []
        for intervention in self.user_profile["intervention_history"]:
            all_strategies.extend(intervention.get("strategies_used", []))
        
        counter = Counter(all_strategies)
        return [f"{strategy}: {count}" for strategy, count in counter.most_common(3)]

    def _analyze_response_patterns(self) -> Dict:
        """Analyze patterns in user responses"""
        if not self.conversation_history:
            return {}
        
        user_messages = [msg["content"] for msg in self.conversation_history 
                        if msg["role"] == "user"]
        
        return {
            "total_messages": len(user_messages),
            "avg_message_length": sum(len(m) for m in user_messages) / len(user_messages),
            "common_themes": self._extract_common_themes(user_messages),
            "engagement_level": "high" if len(user_messages) > 10 else "moderate"
        }

    def _extract_common_themes(self, messages: List[str]) -> List[str]:
        # Simple keyword-based theme extraction
        themes = []
        for topic in self.therapeutic_topics:
            for msg in messages[-10:]:  # Last 10 messages
                if topic in msg.lower():
                    themes.append(topic)
                    break
        return list(set(themes))

    def _create_monitoring_plan(self, assessment: MentalHealthAssessment) -> Dict:
        
        if assessment.risk_level == RiskLevel.CRISIS:
            return {
                "frequency": "Continuous monitoring recommended",
                "checkpoints": ["Immediate", "24 hours", "48 hours", "1 week"],
                "professional_involvement": "Required",
                "safety_planning": "Essential"
            }
        elif assessment.risk_level == RiskLevel.HIGH:
            return {
                "frequency": "Daily check-ins",
                "checkpoints": ["Daily", "3 days", "1 week", "2 weeks"],
                "professional_involvement": "Strongly recommended",
                "safety_planning": "Recommended"
            }
        elif assessment.risk_level == RiskLevel.MODERATE:
            return {
                "frequency": "Every 2-3 days",
                "checkpoints": ["3 days", "1 week", "2 weeks", "1 month"],
                "professional_involvement": "Consider",
                "safety_planning": "Optional"
            }
        else:
            return {
                "frequency": "Weekly check-ins",
                "checkpoints": ["1 week", "2 weeks", "1 month"],
                "professional_involvement": "As needed",
                "safety_planning": "Basic self-monitoring"
            }