# 📡 WSN Enerji Optimizasyonu ve Q-Learning Simülasyonu

Bu proje, Kablosuz Sensör Ağlarında (Wireless Sensor Networks - WSN) veri paketlerinin hedef merkeze (Sink Node) en az enerji harcayarak (en kısa mesafeden) ulaşmasını sağlamak amacıyla **Q-Learning (Pekiştirmeli Öğrenme)** algoritmasının kullanıldığı canlı bir simülasyon ortamıdır.

## 🚀 Projenin Amacı ve Çözdüğü Problem
Sensör ağlarında pillerin bitmesi ağın çökmesine neden olur. Bu projede, veri paketleri rastgele yollar izlemek yerine yapay zeka tarafından eğitilir. 
Ağın tam ortasında aşılmaz bir **coğrafi engel (dağ/göl)** simüle edilmiştir. Yapay zeka, bu engeli fark edip etrafından dolaşarak hedefe giden en verimli ve kesintisiz rotayı kendi kendine keşfeder.

## 🎬 Simülasyon Videosu
Projenin canlı olarak nasıl çalıştığını (yapay zekanın engeli nasıl aştığını) aşağıdaki videodan izleyebilirsiniz:
> **[▶️ Simülasyon Videosunu İzlemek İçin Tıklayın] (https://drive.google.com/drive/u/0/folders/1Cfm42ef-buSxERBr--Q0OkuVo2QpAWJZ)**

---

## 🧠 Kullanılan Teknolojiler ve Algoritma
* **Python 3.8+**: Temel programlama dili.
* **Q-Learning Algoritması**: Veri paketinin rotayı öğrenmesini sağlayan Bellman denklemi tabanlı yapay zeka. (Epsilon-Greedy yaklaşımı kullanılmıştır).
* **NetworkX**: 150 düğümlü (node) sensör ağının matematiksel graf yapısını ve bağlantılarını (edge) kurmak için.
* **Pygame**: Öğrenilen rotanın, veri paketinin zıplayışlarının ve engel aşımının **60 FPS** ile canlı ve görsel olarak izlenmesi için.

## ⚙️ Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

**1. Repoyu Bilgisayarınıza Çekin:**
```bash
git clone [https://github.com/mericsimsek/wsn-qlearning-simulation.git](https://github.com/mericsimsek/wsn-qlearning-simulation.git)
cd wsn-qlearning-simulation
