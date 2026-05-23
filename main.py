import random
import logging
from environment import WSNEnvironment
from q_agent import QRouteOptimizer

if __name__ == "__main__":
    print("\n" + "="*50)
    print("WSN Q-LEARNING ENERJİ OPTİMİZASYON SİMÜLATÖRÜ")
    print("="*50 + "\n")

    # 1. Ağı Kur (70 Düğüm, 25m Menzil)
    env = WSNEnvironment(num_nodes=70, comm_range=25.0)
    env.build_network()

    # 2. Yapay Zekayı Ağa Sal ve Eğit (3000 İterasyon)
    agent = QRouteOptimizer(env, episodes=3000)
    agent.train()

    # 3. Test: Uzak bir sensör seç ve Merkeze nasıl gideceğini izle
    test_node = random.randint(1, env.num_nodes - 1)
    logging.info(f"Test ediliyor: Düğüm {test_node}'den Merkeze (0) rota aranıyor...")
    
    best_route = agent.get_best_route(test_node)
    
    # 4. Sonuçları Değerlendir ve Ekrana Çiz
    if best_route[-1] == 0:
        logging.info(f"Rotayı Buldu! İzlenen Yol: {best_route}")
        env.draw_route(best_route, title=f"Düğüm {test_node}'den Merkeze Q-Learning ile Bulunan En Verimli Rota")
    else:
        logging.warning("Sensör çok ıssız bir yerde, merkeze giden fiziksel bir yol bulamadı! Simülasyonu yeniden başlatın.")
