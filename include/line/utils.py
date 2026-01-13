import torch

class AliasSampler:
    def __init__(self, weights):
        # Using NumPy arrays instead of dictionaries
        n = len(weights)
        self.prob = torch.zeros(n, dtype=torch.float32)
        self.alias = torch.zeros(n, dtype=torch.long)
        sum_w = weights.sum()
        # Construction of scaled probabilities in a single call
        scaled_probs = weights * n / (sum_w if sum_w > 0 else 1.0)
        small, large = [], []

        # Sort the scaled probabilities into their appropriate stacks
        for i, p in enumerate(scaled_probs):
            if p < 1: small.append(i)
            else: large.append(i)
        
        # Construction of probability and alias arrays
        while small and large:
            s, l = small.pop(), large.pop()
            self.prob[s] = scaled_probs[s]
            self.alias[s] = l
            scaled_probs[l] = (scaled_probs[l] + scaled_probs[s]) - 1.0
            if scaled_probs[l] < 1: small.append(l)
            else: large.append(l)
        self.prob[large + small] = 1.0

    def sample(self, size):
        # Selects samples of size 'size' and then, for each sample decides between idx or alias[idx] based on prob[idx]
        idx = torch.randint(0, len(self.prob), (size,))
        rand_vals = torch.rand(size)
        res = torch.where(rand_vals < self.prob[idx], idx, self.alias[idx])
        return res