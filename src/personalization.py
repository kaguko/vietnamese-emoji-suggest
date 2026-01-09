"""
Personalization module with adaptive decay for Vietnamese Emoji Suggestion System.

This module provides:
- User preference tracking
- Exponential decay for recency weighting
- Adaptive emoji ranking based on user history
"""

import math
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class EmojiInteraction:
    """Single emoji interaction record."""
    emoji: str
    emotion: str
    timestamp: str
    selected: bool = True  # True if user selected, False if shown but not selected


@dataclass
class UserPreference:
    """User preference for a specific emoji in an emotion context."""
    emoji: str
    count: int
    last_used: str
    weighted_score: float = 0.0


class AdaptivePersonalizer:
    """
    Adaptive emoji personalizer with exponential decay.
    
    Recent choices matter more than older ones.
    Decay rate controls how quickly old preferences fade.
    """
    
    def __init__(
        self,
        decay_rate: float = 0.1,  # Per day decay
        storage_path: str = "data/user_preferences.json",
        max_history_days: int = 30
    ):
        """
        Initialize personalizer.
        
        Args:
            decay_rate: Exponential decay rate per day (0.1 = 10% decay/day)
            storage_path: Path to store user preferences
            max_history_days: Maximum days to keep history
        """
        self.decay_rate = decay_rate
        self.storage_path = storage_path
        self.max_history_days = max_history_days
        
        # In-memory storage: {user_id: {emotion: {emoji: UserPreference}}}
        self.preferences: Dict[str, Dict[str, Dict[str, UserPreference]]] = {}
        
        # Interaction history: {user_id: [EmojiInteraction]}
        self.history: Dict[str, List[EmojiInteraction]] = {}
        
        # Load existing data
        self._load_preferences()
    
    def _load_preferences(self):
        """Load preferences from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.preferences = data.get('preferences', {})
                    self.history = data.get('history', {})
            except (json.JSONDecodeError, IOError):
                self.preferences = {}
                self.history = {}
    
    def _save_preferences(self):
        """Save preferences to storage."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump({
                'preferences': self.preferences,
                'history': self.history,
                'last_updated': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def record_selection(
        self,
        user_id: str,
        emoji: str,
        emotion: str,
        was_selected: bool = True
    ):
        """
        Record user's emoji selection.
        
        Args:
            user_id: User identifier
            emoji: Emoji that was shown/selected
            emotion: Detected emotion context
            was_selected: True if user selected this emoji
        """
        now = datetime.now().isoformat()
        
        # Record interaction
        interaction = EmojiInteraction(
            emoji=emoji,
            emotion=emotion,
            timestamp=now,
            selected=was_selected
        )
        
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id].append(asdict(interaction))
        
        # Update preferences if selected
        if was_selected:
            if user_id not in self.preferences:
                self.preferences[user_id] = {}
            if emotion not in self.preferences[user_id]:
                self.preferences[user_id][emotion] = {}
            
            if emoji not in self.preferences[user_id][emotion]:
                self.preferences[user_id][emotion][emoji] = {
                    'emoji': emoji,
                    'count': 0,
                    'last_used': now,
                    'weighted_score': 0.0
                }
            
            pref = self.preferences[user_id][emotion][emoji]
            pref['count'] += 1
            pref['last_used'] = now
        
        # Periodically save
        if len(self.history.get(user_id, [])) % 5 == 0:
            self._save_preferences()
    
    def _calculate_decay_weight(self, last_used: str) -> float:
        """
        Calculate decay weight based on time since last use.
        
        Uses exponential decay: weight = e^(-days * decay_rate)
        
        Args:
            last_used: ISO timestamp of last use
            
        Returns:
            Weight between 0 and 1
        """
        try:
            last_dt = datetime.fromisoformat(last_used)
            now = datetime.now()
            days_ago = (now - last_dt).days
            
            # Exponential decay
            weight = math.exp(-days_ago * self.decay_rate)
            return max(0.01, min(1.0, weight))  # Clamp to [0.01, 1.0]
        except (ValueError, TypeError):
            return 0.5  # Default weight if parsing fails
    
    def get_user_preferences(
        self,
        user_id: str,
        emotion: str
    ) -> List[Tuple[str, float]]:
        """
        Get user's weighted emoji preferences for an emotion.
        
        Args:
            user_id: User identifier
            emotion: Emotion context
            
        Returns:
            List of (emoji, weighted_score) tuples, sorted by score
        """
        if user_id not in self.preferences:
            return []
        if emotion not in self.preferences[user_id]:
            return []
        
        scored_prefs = []
        
        for emoji, pref in self.preferences[user_id][emotion].items():
            # Calculate weighted score with decay
            decay_weight = self._calculate_decay_weight(pref['last_used'])
            weighted_score = pref['count'] * decay_weight
            scored_prefs.append((emoji, weighted_score))
        
        # Sort by weighted score (descending)
        scored_prefs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_prefs
    
    def rank_emojis(
        self,
        user_id: str,
        emotion: str,
        base_emojis: List[str],
        personalization_weight: float = 0.4
    ) -> List[str]:
        """
        Rank emojis combining base suggestions with user preferences.
        
        Args:
            user_id: User identifier
            emotion: Detected emotion
            base_emojis: Base emoji suggestions from model
            personalization_weight: Weight for personalization (0-1)
            
        Returns:
            Re-ranked list of emojis
        """
        # Get user preferences
        user_prefs = self.get_user_preferences(user_id, emotion)
        
        if not user_prefs:
            # No preferences yet, return base suggestions
            return base_emojis
        
        # Create score dict
        emoji_scores = {}
        
        # Base scores (position-weighted)
        base_weight = 1.0 - personalization_weight
        for i, emoji in enumerate(base_emojis):
            position_score = (len(base_emojis) - i) / len(base_emojis)
            emoji_scores[emoji] = base_weight * position_score
        
        # Add personalization scores
        max_pref_score = max(score for _, score in user_prefs) if user_prefs else 1
        for emoji, pref_score in user_prefs:
            normalized_score = pref_score / max_pref_score if max_pref_score > 0 else 0
            if emoji in emoji_scores:
                emoji_scores[emoji] += personalization_weight * normalized_score
            else:
                # User preferred emoji not in base suggestions
                emoji_scores[emoji] = personalization_weight * normalized_score * 0.8
        
        # Sort by score
        ranked = sorted(emoji_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top emojis (same count as base)
        return [emoji for emoji, _ in ranked[:len(base_emojis)]]
    
    def get_user_stats(self, user_id: str) -> Dict:
        """
        Get user statistics.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with user statistics
        """
        if user_id not in self.preferences:
            return {
                'user_id': user_id,
                'total_interactions': 0,
                'emotions_used': [],
                'favorite_emojis': [],
                'active_days': 0
            }
        
        total_interactions = 0
        emotions_used = list(self.preferences[user_id].keys())
        all_emojis = []
        
        for emotion, emojis in self.preferences[user_id].items():
            for emoji, pref in emojis.items():
                total_interactions += pref['count']
                all_emojis.append((emoji, pref['count']))
        
        # Top emojis
        all_emojis.sort(key=lambda x: x[1], reverse=True)
        favorite_emojis = [e for e, _ in all_emojis[:5]]
        
        # Active days (from history)
        active_dates = set()
        for interaction in self.history.get(user_id, []):
            try:
                dt = datetime.fromisoformat(interaction['timestamp'])
                active_dates.add(dt.date())
            except (ValueError, KeyError):
                pass
        
        return {
            'user_id': user_id,
            'total_interactions': total_interactions,
            'emotions_used': emotions_used,
            'favorite_emojis': favorite_emojis,
            'active_days': len(active_dates)
        }
    
    def get_emotion_history(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Dict[str, int]]:
        """
        Get emotion history for visualization.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            Dict of {date: {emotion: count}}
        """
        history = {}
        cutoff = datetime.now() - timedelta(days=days)
        
        for interaction in self.history.get(user_id, []):
            try:
                dt = datetime.fromisoformat(interaction['timestamp'])
                if dt >= cutoff:
                    date_str = dt.strftime('%Y-%m-%d')
                    emotion = interaction['emotion']
                    
                    if date_str not in history:
                        history[date_str] = {}
                    history[date_str][emotion] = history[date_str].get(emotion, 0) + 1
            except (ValueError, KeyError):
                pass
        
        return history
    
    def cleanup_old_data(self):
        """Remove data older than max_history_days."""
        cutoff = datetime.now() - timedelta(days=self.max_history_days)
        
        for user_id in list(self.history.keys()):
            self.history[user_id] = [
                interaction for interaction in self.history[user_id]
                if datetime.fromisoformat(interaction['timestamp']) >= cutoff
            ]
        
        self._save_preferences()
    
    def reset_user(self, user_id: str):
        """Reset all data for a user."""
        if user_id in self.preferences:
            del self.preferences[user_id]
        if user_id in self.history:
            del self.history[user_id]
        self._save_preferences()


# Singleton instance for app-wide use
_personalizer_instance = None

def get_personalizer() -> AdaptivePersonalizer:
    """Get singleton personalizer instance."""
    global _personalizer_instance
    if _personalizer_instance is None:
        _personalizer_instance = AdaptivePersonalizer()
    return _personalizer_instance


if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE PERSONALIZATION TEST")
    print("=" * 60)
    
    # Test personalization
    personalizer = AdaptivePersonalizer(
        decay_rate=0.1,
        storage_path="data/test_preferences.json"
    )
    
    # Simulate user interactions
    user_id = "test_user_001"
    
    # Record some selections
    print("\n1. Recording user selections...")
    personalizer.record_selection(user_id, "😊", "joy", True)
    personalizer.record_selection(user_id, "🎉", "joy", True)
    personalizer.record_selection(user_id, "😊", "joy", True)  # Prefers 😊
    personalizer.record_selection(user_id, "😢", "sadness", True)
    
    # Get preferences
    print("\n2. User preferences for 'joy':")
    prefs = personalizer.get_user_preferences(user_id, "joy")
    for emoji, score in prefs:
        print(f"   {emoji}: {score:.2f}")
    
    # Rank emojis
    print("\n3. Ranking emojis with personalization:")
    base_emojis = ["🥳", "🎊", "✨"]  # Model suggestions
    ranked = personalizer.rank_emojis(user_id, "joy", base_emojis)
    print(f"   Base: {base_emojis}")
    print(f"   Ranked: {ranked}")
    
    # User stats
    print("\n4. User statistics:")
    stats = personalizer.get_user_stats(user_id)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✓ Personalization test complete!")
