import numpy as np
import random
import logging

class QRouteOptimizer:
    """Yönlendirme için Q-Learning Ajanı Sınıfı"""
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, episodes=2000):
        self.env = env
        self.alpha = alpha                # Öğrenme Oranı
        self.gamma = gamma                # Gelecekteki ödül tahmini çarpanı
        self.epsilon = epsilon            # Keşif yapma oranı
        self.epsilon_decay = 0.995        # Keşfi zamanla azaltma
        self.episodes = episodes
        self.q_table = np.zeros((env.num_nodes, env.num_nodes))

    def train(self):
        logging.info("Q-Learning Ajanı Eğitiliyor... Lütfen bekleyin.")
        for episode in range(self.episodes):
            state = random.randint(1, self.env.num_nodes - 1)
            step = 0
            
            while state != self.env.base_station_id and step < self.env.num_nodes:
                neighbors = list(self.env.graph.neighbors(state))
                if not neighbors: break # Çıkmaz sokak
                
                # Epsilon-Greedy ile komşu seç (Keşif veya Sömürü)
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(neighbors)
                else:
                    q_values = [self.q_table[state, n] for n in neighbors]
                    action = neighbors[np.argmax(q_values)]

                # Ödül veya Ceza Hesapla (-1 enerji harcama cezası, +100 hedefe ulaşma ödülü)
                reward = 100.0 if action == self.env.base_station_id else -1.0
                
                # Bellman Denklemi ile Q-Table Güncelle
                next_neighbors = list(self.env.graph.neighbors(action))
                max_next_q = max([self.q_table[action, n] for n in next_neighbors]) if next_neighbors else 0.0
                
                self.q_table[state, action] = self.q_table[state, action] + self.alpha * (
                    reward + self.gamma * max_next_q - self.q_table[state, action]
                )
                
                state = action
                step += 1
                
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        logging.info("Eğitim Tamamlandı! Ağ en verimli yolları öğrendi.")

    def get_best_route(self, start_node):
        """Öğrenilmiş Q-Table üzerinden en mükemmel rotayı çıkartır"""
        route = [start_node]
        current = start_node
        visited = {start_node}
        
        while current != self.env.base_station_id:
            neighbors = list(self.env.graph.neighbors(current))
            if not neighbors: break
            
            # Gidilmemiş komşular arasında en yüksek Q puanına sahip olanı seç
            best_n = None
            max_q = -float('inf')
            for n in neighbors:
                if n not in visited and self.q_table[current, n] > max_q:
                    max_q = self.q_table[current, n]
                    best_n = n
                    
            if best_n is None: break
            
            route.append(best_n)
            visited.add(best_n)
            current = best_n
            
        return route

