class SGD:
    def __init__(self, tempo=0.01, ped=0.0): self.tempo, self.ped = tempo, ped
    def krok(self, parametry):
        for p in parametry:
            if p.gradient is not None: p.dane -= self.tempo * (p.gradient + self.ped * p.dane)
    def wyzeruj_gradient(self, parametry):
        for p in parametry: p.wyzeruj_gradient()


class Adam(SGD):
    def __init__(self, tempo=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(tempo); self.beta1, self.beta2, self.epsilon, self.t = beta1, beta2, epsilon, 0; self.m, self.v = {}, {}
    def krok(self, parametry):
        self.t += 1
        for p in parametry:
            if p.gradient is None: continue
            xp = p.modul; key = id(p); self.m[key] = self.beta1 * self.m.get(key, xp.zeros_like(p.dane)) + (1 - self.beta1) * p.gradient
            self.v[key] = self.beta2 * self.v.get(key, xp.zeros_like(p.dane)) + (1 - self.beta2) * p.gradient ** 2
            m = self.m[key] / (1 - self.beta1 ** self.t); v = self.v[key] / (1 - self.beta2 ** self.t)
            p.dane -= self.tempo * m / (xp.sqrt(v) + self.epsilon)
