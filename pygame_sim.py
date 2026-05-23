import pygame
import random
import math
import numpy as np
import networkx as nx
import time

class WSNEnvironment:
    def __init__(self, num_nodes=150, comm_range=20.0, area_size=100.0):
        self.num_nodes = num_nodes
        self.comm_range = comm_range
        self.area_size = area_size
        self.base_station_id = 0
        self.graph = nx.Graph()
        self.pos = {}

    def build_network(self):
        # Merkez İstasyonu (Hedefi) haritanın en soluna koyuyoruz
        self.pos[0] = (10, 50)
        self.graph.add_node(0)
        
        for i in range(1, self.num_nodes):
            self.graph.add_node(i)
            while True:
                x = np.random.uniform(0, self.area_size)
                y = np.random.uniform(0, self.area_size)
                
                # --- KARMAŞIK TOPOLOJİ: ORTAYA DEVASA BİR ENGEL (DAĞ/GÖL) KOY ---
                # X: 40 ile 60 arası, Y: 20 ile 80 arası boş kalacak. 
                # Yapay zeka düz gidemeyecek, etrafından dolanmak ZORUNDA.
                if 40 < x < 60 and 20 < y < 80:
                    continue 
                
                self.pos[i] = (round(x, 2), round(y, 2))
                break

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                x1, y1 = self.pos[i]
                x2, y2 = self.pos[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist <= self.comm_range:
                    self.graph.add_edge(i, j, weight=round(dist, 2))

class QRouteOptimizer:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, episodes=4000): # Engel olduğu için eğitimi 4000'e çıkardık
        self.env = env
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.epsilon_decay = 0.995
        self.episodes = episodes
        self.q_table = np.zeros((env.num_nodes, env.num_nodes))

    def train(self):
        print("\n" + "="*60)
        print("  [YAPAY ZEKA] ZORLU HARİTADA Q-LEARNING EĞİTİMİ BAŞLADI")
        print("="*60)
        start_time = time.time()
        
        for episode in range(self.episodes):
            state = random.randint(1, self.env.num_nodes - 1)
            step = 0
            while state != self.env.base_station_id and step < self.env.num_nodes:
                neighbors = list(self.env.graph.neighbors(state))
                if not neighbors: break
                
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(neighbors)
                else:
                    q_values = [self.q_table[state, n] for n in neighbors]
                    action = neighbors[np.argmax(q_values)]

                reward = 100.0 if action == self.env.base_station_id else -1.0
                next_neighbors = list(self.env.graph.neighbors(action))
                max_next_q = max([self.q_table[action, n] for n in next_neighbors]) if next_neighbors else 0.0
                
                self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward + self.gamma * max_next_q - self.q_table[state, action])
                state = action; step += 1
            
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 1000 == 0:
                print(f" [*] İterasyon {episode + 1}/{self.episodes} | Q-Matrisi engelleri aşıyor...")

        print(f"\n[BAŞARILI] Eğitim {(time.time() - start_time):.2f} saniyede bitti.")

    def get_best_route(self, start_node):
        route = [start_node]
        current, visited = start_node, {start_node}
        total_distance = 0.0
        
        while current != self.env.base_station_id:
            neighbors = list(self.env.graph.neighbors(current))
            if not neighbors: break
            best_n, max_q = None, -float('inf')
            for n in neighbors:
                if n not in visited and self.q_table[current, n] > max_q:
                    max_q = self.q_table[current, n]
                    best_n = n
            if best_n is None: break
            
            x1, y1 = self.env.pos[current]
            x2, y2 = self.env.pos[best_n]
            total_distance += math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            route.append(best_n); visited.add(best_n); current = best_n
            
        return route, total_distance

def run_live_simulation():
    env = WSNEnvironment(num_nodes=150, comm_range=20.0)
    env.build_network()

    agent = QRouteOptimizer(env, episodes=4000)
    agent.train()

    # Şovu garantilemek için özellikle haritanın EN SAĞINDAN (engelin arkasından) bir düğüm seçiyoruz
    far_right_nodes = [n for n in env.graph.nodes() if env.pos[n][0] > 80]
    if far_right_nodes:
        test_node = random.choice(far_right_nodes)
    else:
        test_node = random.randint(1, env.num_nodes - 1)

    print(f"\n[ZORLU TEST] Engelin arkasındaki Düğüm {test_node}'den Merkeze yol aranıyor...")
    best_route, total_dist = agent.get_best_route(test_node)
    
    if best_route[-1] != 0:
        print(f"\n[UYARI] {test_node} numaralı düğüm çok ıssız, lütfen scripti tekrar çalıştırın.")
        return

    print(f"\n[SONUÇ] Yapay Zeka Engeli Aşarak Rotayı Buldu!")
    print(f" > İZLENEN DÜĞÜM SIRASI: {best_route}")
    
    print("\n>>> Pygame Simülasyon Ekranı Açılıyor...")
    time.sleep(2)

    pygame.init()
    pygame.font.init() # Font motorunu başlat
    font = pygame.font.SysFont("arial", 12, bold=True) # Düğüm numaraları için font

    WIDTH, HEIGHT = 800, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("WSN Q-Learning - Engelli Arazi & Düğüm ID'leri")
    clock = pygame.time.Clock()

    BG_COLOR = (20, 20, 25)
    EDGE_COLOR = (40, 40, 50)
    NODE_COLOR = (180, 220, 255) # Açık mavi düğümler (Üstündeki siyah yazı okunsun diye)
    SINK_COLOR = (255, 100, 100)
    ROUTE_COLOR = (0, 255, 100)
    PACKET_COLOR = (255, 255, 0)

    def scale_pos(pos):
        margin = 50
        scale = (WIDTH - 2 * margin) / env.area_size
        return (int(pos[0] * scale + margin), int(pos[1] * scale + margin))

    running = True
    packet_route_index = 0
    packet_progress = 0.0 

    while running:
        screen.fill(BG_COLOR)
        
        # 1. Zayıf bağlantıları çiz
        for u, v in env.graph.edges():
            pygame.draw.line(screen, EDGE_COLOR, scale_pos(env.pos[u]), scale_pos(env.pos[v]), 1)

        # 2. Kalın Yeşil Rotayı Çiz
        for i in range(len(best_route) - 1):
            pygame.draw.line(screen, ROUTE_COLOR, scale_pos(env.pos[best_route[i]]), scale_pos(env.pos[best_route[i+1]]), 4)

        # 3. Tüm Düğümleri ve Üzerlerindeki Numaraları Çiz
        for node in env.graph.nodes():
            color = SINK_COLOR if node == env.base_station_id else NODE_COLOR
            size = 14 if node == env.base_station_id else 11 # Numaralar sığsın diye biraz büyüttük
            center = scale_pos(env.pos[node])
            
            # Yuvarlağı çiz
            pygame.draw.circle(screen, color, center, size)
            
            # İçine Numarayı (ID) yazdır
            text_surface = font.render(str(node), True, (10, 10, 10)) # Siyah yazı
            text_rect = text_surface.get_rect(center=center)
            screen.blit(text_surface, text_rect)

        # 4. Veri Paketini Hareket Ettir
        if packet_route_index < len(best_route) - 1:
            start_pos = np.array(scale_pos(env.pos[best_route[packet_route_index]]))
            end_pos = np.array(scale_pos(env.pos[best_route[packet_route_index + 1]]))
            
            current_pos = start_pos + (end_pos - start_pos) * packet_progress
            pygame.draw.circle(screen, PACKET_COLOR, (int(current_pos[0]), int(current_pos[1])), 8)
            
            packet_progress += 0.015 # Biraz daha yavaş aksın ki numaralar rahat okunsun
            
            if packet_progress >= 1.0:
                packet_progress = 0.0
                packet_route_index += 1
        else:
            target = scale_pos(env.pos[env.base_station_id])
            pygame.draw.circle(screen, PACKET_COLOR, target, 18, 3)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    run_live_simulation()
