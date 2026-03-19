import random
from parameters import decay, TIMESTEP_TIME, CHORD_LEARNING_RATE, CHORD_INITIAL_STRENGTH


class ChordMemory:
    """
    Tracks per-chord memory strength for a set of chord shortcuts.

    Each chord starts unknown (strength=0). Strength grows with use and decays
    over time using the same exponential decay function as the working memory
    model (tied to memory_param / parameters[0]).

    Recall is probabilistic: P(recall | word) = strength[word].
    """

    def __init__(self, chord_dict, memory_param=0.5, initial_strength=None):
        self.chord_dict = chord_dict        # {word: (left_key, right_key)}
        self.memory_param = memory_param
        self._initial_strength = initial_strength if initial_strength is not None else CHORD_INITIAL_STRENGTH
        self.strength = {w: self._initial_strength for w in chord_dict}

    def scan(self, remaining_text):
        """Return the longest chord word that matches the start of remaining_text, or None."""
        best = None
        for word in self.chord_dict:
            if remaining_text.startswith(word):
                if best is None or len(word) > len(best):
                    best = word
        return best

    def try_recall(self, word):
        """Probabilistic recall: returns True with probability equal to current strength."""
        return random.random() < self.strength[word]

    def use(self, word):
        """Boost strength after a successful chord use (simulates reinforcement learning)."""
        self.strength[word] = min(1.0, self.strength[word] + CHORD_LEARNING_RATE)

    def decay_step(self):
        """Decay all chord strengths by one timestep. Call once per env step."""
        for word in self.strength:
            self.strength[word] *= decay(TIMESTEP_TIME, self.memory_param)

    def reset(self, memory_param=None, initial_strength=None):
        """Reset all strengths to initial value, optionally updating memory_param."""
        if memory_param is not None:
            self.memory_param = memory_param
        if initial_strength is not None:
            self._initial_strength = initial_strength
        self.strength = {w: self._initial_strength for w in self.chord_dict}

    def strengths_snapshot(self):
        """Return a copy of the current strength dict (for logging/rendering)."""
        return dict(self.strength)
