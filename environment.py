import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class WSNEnvironment:
    """Kablosuz Sensör Ağı (WSN) simülasyon ortamı"""
    def __init__(self, num_nodes=50, comm_range=25.0, area_size=100.0):
        self.num_nodes = num_nodes
        self.comm_range = comm_range
        self.area_size = area_size
        self.base_station_id = 0  # 0 numaralı düğüm her zaman Hedef Merkezdir
        self.graph = nx.Graph()
        self.pos = {}
        logging.info(f"WSN Ortamı: {num_nodes} düğüm, {comm_range}m iletişim menzili.")

    def build_network(self):
        """Sensörleri alana dağıtır ve menzile göre birbirine bağlar."""
        for i in range(self.num_nodes):
            self.graph.add_node(i)
            if i == self.base_station_id:
                self.pos[i] = (self.area_size / 2, self.area_size / 2) # Merkez tam ortada
            else:
                self.pos[i] = (np.random.uniform(0, self.area_size), np.random.uniform(0, self.area_size))

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                x1, y1 = self.pos[i]
                x2, y2 = self.pos[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist <= self.comm_range:
                    self.graph.add_edge(i, j, weight=dist)

    def draw_route(self, route, title="Ağ Haritası (Canlı Simülasyon)"):
        """Eğitimden sonra bulunan en iyi rotayı adım adım animasyonla çizer"""
        plt.ion()  # İnteraktif modu aç (Canlı çizim için)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Başlangıç durumu (sadece gri sensörler ve kırmızı merkez)
        node_colors = ['red' if n == self.base_station_id else 'lightgray' for n in self.graph.nodes()]
        
        nx.draw(self.graph, self.pos, ax=ax, node_color=node_colors, node_size=300, 
                with_labels=True, font_size=8, edge_color='whitesmoke')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.draw()
        plt.pause(1.0) # Ağı göster ve paketin yola çıkması için 1 saniye bekle
        
        # Veri paketinin adım adım ilerleyişi
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i+1]
            
            # Geçilen düğümü yeşile boya
            if current_node != self.base_station_id:
                node_colors[current_node] = 'limegreen'
                
            # Rotayı yeşil çizgiyle birleştir
            nx.draw_networkx_edges(self.graph, self.pos, ax=ax, 
                                   edgelist=[(current_node, next_node)], 
                                   edge_color='green', width=3.0)
            
            # Düğümün yeni rengini ekrana bas
            nx.draw_networkx_nodes(self.graph, self.pos, ax=ax, 
                                   nodelist=[current_node], 
                                   node_color='limegreen', node_size=300)
            
            plt.draw()
            plt.pause(0.5) # Her sıçramada yarım saniye bekle (Şov kısmı)
            
        plt.ioff() # İnteraktif modu kapat
        plt.show() # İşlem bitince ekranın kapanmasını engelle
