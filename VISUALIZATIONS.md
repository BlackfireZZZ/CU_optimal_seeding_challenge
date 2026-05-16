# Визуализации: наблюдения и выводы

## Обзор

Сгенерировано 15 статических визуализаций (PNG) и 1 интерактивная (HTML) на основе данных графа (3953 узла, 84070 ребер) и результатов анализа из `deep_analysis_data.json`.

Все файлы в папке `viz/`.

---

## 1. Degree Distribution (`01_degree_distribution.png`)

**Что видно:** Степенное распределение (power-law) с длинным хвостом. Медиана степени = 25, среднее = 42.5 — правый хвост из нескольких хабов (до 293) сильно тянет среднее.

**Вывод:** Граф scale-free. Хабы дорогие для seeding (cost = 300 * degree), но не обязательно эффективные — большинство из них не запускают каскад вообще.

---

## 2. Centrality Scatter Matrix (`02_centrality_scatter_matrix.png`)

**Что видно:** Попарные scatter-plot 5 центральностей (Degree, Betweenness, Closeness, PageRank, K-core). Degree и K-core сильно коррелируют. Betweenness образует характерный "веер" — узлы с одинаковым degree могут иметь betweenness от 0 до 0.17.

**Вывод:** Нельзя полагаться на одну метрику. Betweenness выявляет принципиально другие узлы, чем Degree/K-core.

---

## 3. Community Analysis (`03_community_analysis.png`)

**Что видно:**
- **Cascade efficiency:** Community C9 (512), C2 (416), C4 (240), C28 (180), C37 (135) имеют viral/size > 0.95 — каскад покрывает почти всю community.
- **Провальные:** C15 (393 узла), C17 (323), C1 (277) — viral/size < 0.05. Каскад туда практически не проникает при одиночном seeding.
- **ROI:** Все community имеют отрицательный ROI при полном seeding (seed cost > viral income), но C22 (-0.59) и C37 (-0.73) ближе к break-even.
- **Min seeds:** Маленькие community (C37, C22, C5) нужен всего 1 seed для 90% каскада. Крупные (C15, C17, C1) требуют 8 seeds, но даже тогда каскад слабый.

**Вывод:** Стратегия должна фокусироваться на "каскадабельных" community (C9, C2, C4, C37) и игнорировать C15/C17/C1 — они просто сжигают бюджет.

---

## 4. Positive-Profit Seeds (`04_positive_profit_seeds.png`)

**Что видно:**
- **Лидеры:** Node 1304 (profit 32000, viral 862, degree 37) и node 3057 (profit 30200, viral 700, degree 16) — абсолютные чемпионы.
- **Community C37** содержит кластер из ~10 прибыльных seeds с profit 4500-9000. Любой из них запускает каскад на 216 узлов (вся community).
- **ROI:** Лучший ROI у узлов с low degree (6-16) в маленьких community — cost минимальный, а каскад покрывает всю community.

**Вывод:** Node 3057 — лучший по ROI (profit/cost = 6.3), node 1304 — лучший по абсолютному profit. Оптимальная стратегия: сначала дешевые seeds из C37, потом 3057 и 1304.

---

## 5. K-Core Decomposition (`05_kcore_decomposition.png`)

**Что видно:**
- Максимальный k-core = 114 (157 узлов). Это плотное ядро графа.
- Кумулятивная кривая показывает скачок на k=20-40 — большинство узлов в средних core-слоях.
- Degree и K-core сильно коррелируют (r=0.93), но есть outlier-ы: узлы с degree 50-80 в core 20-40 (периферия плотных community).

**Вывод:** K-core 114 — дорогое и неэффективное ядро для seeding (все узлы high-degree, cascade не запускается из-за высокого порога 18%). Для задачи influence maximization k-core — плохой предиктор.

---

## 6. Cascade Dynamics (`06_cascade_dynamics.png`)

**Что видно:**
- **Threshold:** При degree 1-5 нужен всего 1 инфицированный сосед. При degree 50 — уже 9 соседей.
- **Difficulty distribution:** 17% узлов "легкие" (need 1), 20% средние (need 2), 15% — need 3, и 49% — "тяжелые" (need 8+).
- **Stacked histogram:** Тяжелые узлы (красные) доминируют в high-degree диапазоне (>30).

**Вывод:** Каскад легко распространяется по low-degree периферии, но застревает на хабах. Стратегия: seed в community с большим количеством low-degree узлов.

---

## 7. CI vs PageRank vs Degree (`07_ci_pagerank_degree.png`)

**Что видно:**
- **CI vs Degree:** Суперлинейная зависимость. Nodes 2347 и 2543 (degree ~290) имеют CI ~12M — они в самом плотном ядре.
- **CI-ROI vs PR-ROI:** Всего 2 пересечения в top-30 — метрики выделяют разные узлы.
- **PageRank distribution:** Log-normal, основная масса 0.0001-0.0004.

**Вывод:** CI и PageRank оптимизируют разные свойства. CI лучше для LT-модели (учитывает структуру окрестности), но top CI nodes — это дорогие хабы, неэффективные по ROI.

---

## 8. Cost-Effectiveness (`08_cost_effectiveness.png`)

**Что видно:**
- Подавляющее большинство узлов (из top-200 scored) имеют viral=0 — их seeding не запускает каскад вообще.
- Прибыльные seeds (красные звезды) сконцентрированы в low-cost зоне.
- Pareto frontier крутая: резкий переход от viral=0 к viral=700+ при cost ~5000.

**Вывод:** В графе есть четкий "фазовый переход" — либо seed запускает каскад (и он выгодный), либо нет. Промежуточных случаев мало.

---

## 9. Network Communities (`09_network_communities.png`)

**Что видно:**
- Community четко разделены пространственно (spring layout). Видны ~20 плотных кластеров.
- Articulation points (красные кольца, 86 шт.) сосредоточены на границах community — мосты между группами.
- Profit seeds (желтые звезды) расположены внутри компактных community, а не на границах.

**Вывод:** Прибыльные seeds работают за счет каскада ВНУТРИ своей community. Мостовые узлы (articulation points) не являются хорошими seeds — они соединяют community, но не запускают каскад ни в одной из них.

---

## 10. Summary Dashboard (`10_summary_dashboard.png`)

Сводная панель всех ключевых метрик в одном изображении. Удобно для быстрого обзора.

---

## 11. Centrality Correlation Heatmap (`11_centrality_correlation_heatmap.png`)

**Что видно:**
| Пара | Корреляция | Интерпретация |
|------|-----------|---------------|
| Degree - K-core | **0.927** | Почти одно и то же |
| Degree - Eigenvector | **0.631** | Хабы = eigenvector-центры |
| Degree - PageRank | 0.611 | Умеренная связь |
| Degree - Closeness | 0.426 | Слабая связь |
| Degree - Betweenness | **0.113** | Почти нет связи |
| Betweenness - Eigenvector | **-0.016** | Отрицательная! |
| Betweenness - K-core | 0.037 | Нет связи |

**Вывод:** Betweenness centrality измеряет принципиально другое свойство графа. Высоко-betweenness узлы — это мосты между community, они часто имеют средний degree. Eigenvector и K-core сильно коррелированы с Degree (>0.63) и могут быть redundant.

---

## 12. Cost-Effectiveness Fixed (`12_cost_effectiveness_fixed.png`)

**Что видно:** Только узлы с viral > 0 (из scored nodes). Левый график: positive-profit seeds отмечены красными кольцами. Node 1304 на вершине (862 viral), 3057 и 1505 чуть ниже (700 viral).

**Вывод:** Из 200 scored nodes только ~35 вообще запускают каскад. Выбор seed — это в первую очередь бинарный вопрос "каскад или нет", а не градиент.

---

## 13. Community Meta-Graph (`13_community_metagraph.png`)

**Что видно:**
- Центральные community (C7, C15, C1) — крупные, но с низкой viral efficiency (красные/темные узлы).
- Периферийные community (C37, C22, C5) — маленькие, но зеленые (высокая viral efficiency).
- Толстые ребра между C7-C15, C7-C1 — много cross-edges, но каскад все равно не распространяется.

**Вывод:** Между "трудными" community много связей, но это не помогает каскаду — порог 18% слишком высок для cross-community распространения. Каждую community нужно seeding отдельно.

---

## 14. Degree vs Betweenness (`14_degree_vs_betweenness.png`)

**Что видно:**
- **Outlier-мосты:** Node 1085 (degree ~65, betweenness 0.175) — главный мост графа. Node 1718 (degree ~135, betweenness 0.148).
- **Bridge nodes** (правый график, zoom): десятки узлов с degree 10-60 и betweenness 0.01-0.07 — межкоммунитарные мосты.
- Основная масса узлов имеет betweenness < 0.005.

**Вывод:** Мостовые узлы интересны для фазы 2 стратегии (распространение между community), но не для начального seeding.

---

## 15. Multi-Centrality Rankings (`15_multi_centrality_rankings.png`)

**Что видно:**
- Nodes 2543, 2347 стабильно в top-5 по Degree, K-core, Eigenvector — но проваливаются по Betweenness.
- Node 1085 лидер по Betweenness, но средний по остальным.
- Линии сильно пересекаются между Betweenness и остальными метриками — подтверждает ортогональность.

**Вывод:** Нет "универсально лучшего" узла. Для influence maximization нужна композитная метрика, а лучше — прямая симуляция каскада (как в solution.py).

---

## Интерактивная сеть (`network_interactive.html`)

Plotly-визуализация giant component. При наведении на узел видны: degree, k-core, community, betweenness, pagerank. Profit seeds отмечены желтыми звездами.

---

## Общие выводы

1. **Betweenness — ортогональная метрика.** Корреляция с Degree всего 0.113. Мостовые узлы — отдельный класс, полезный для межкоммунитарного распространения, но не для запуска каскада.

2. **Каскад бинарен.** Узел либо запускает каскад (и дает прибыль), либо нет. Из 3953 узлов только 33 прибыльны при одиночном seeding.

3. **Community определяет все.** Прибыльные seeds работают внутри "каскадабельных" community. Cross-community распространение при пороге 18% почти не работает.

4. **Low-degree seeds лучше.** Node 3057 (degree 16, profit 30200) эффективнее node 2543 (degree 293, profit -87900). ROI обратно пропорционален degree.

5. **K-core бесполезен для этой задачи.** Плотное ядро (k=114) содержит дорогие узлы, где каскад не запускается из-за высокого порога активации.

6. **Оптимальная стратегия:** seed в маленькие "каскадабельные" community (C37, C22, C8) дешевыми узлами, затем 3057 и 1304 для крупных community (C9, C17).
