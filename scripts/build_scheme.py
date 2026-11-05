"""One maintained source for the three-sheet search scheme and its SVG previews.

Why one description: two hand-laid generators drifted apart. The scheme is described here once and rendered twice.
Edit this description, then regenerate docs/scheme.drawio, scheme.svg, scheme_decision.svg and scheme_failures.svg.
The draw.io file remains viewable/editable in diagrams.net, but manual edits are not the maintained source.
The overwrite flag prevents accidental replacement of the generated artifacts.

Run (only to regenerate from this description): python3 -I scripts/build_scheme.py --overwrite
"""
from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

DOCS = Path(__file__).resolve().parent.parent / "docs"

# fill, stroke, font colour, dashed
COLORS = {
    "code": ("#e8f1ff", "#3b6fd4", "#12315f", 0),
    "model": ("#e3f5e8", "#2f9e57", "#0f4a26", 0),
    "mixed": ("#e8f1ff|#e3f5e8", "#3b6fd4", "#12315f", 0),          # code and a model in one step: a gradient
    "human": ("#fff0dc", "#e08a1e", "#5c3500", 0),
    "humanNI": ("#fff0dc", "#e08a1e", "#5c3500", 1),
    "notimpl": ("#f3f3f3", "#8a8a8a", "#444444", 1),
    "user": ("#ffffff", "#222222", "#111111", 0),
    "same": ("#e3f4e8", "#2f9e57", "#0f4a26", 0),
    "substitute": ("#e4eefc", "#3b6fd4", "#12315f", 0),
    "check": ("#fdf0d4", "#d08a1a", "#5c3500", 0),
    "reject": ("#fbe4e4", "#c43b3b", "#6b1414", 0),
    "question": ("#ffffff", "#444444", "#111111", 0),
}
TAG = {"словарь": "#e08a1e", "правила": "#a35c00"}


class Page:
    def __init__(self, name: str, width: int, height: int):
        self.name, self.width, self.height = name, width, height
        self.nodes: dict[str, dict] = {}
        self.items: list[tuple] = []            # drawing order: ("band"|"node"|"text"|"edge", payload)

    def band(self, x, y, w, h, fill, stroke, title, font):
        self.items.append(("band", dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, title=title, font=font)))

    def node(self, key, x, y, w, h, cls, title, lines=(), count=None, shape="rect", tags=(), size=13):
        n = dict(key=key, x=x, y=y, w=w, h=h, cls=cls, title=title, lines=list(lines), count=count, shape=shape, tags=list(tags), size=size)
        self.nodes[key] = n
        self.items.append(("node", n))

    def text(self, x, y, w, h, lines, size=13, bold=False, color="#111111", align="left"):
        self.items.append(("text", dict(x=x, y=y, w=w, h=h, lines=[lines] if isinstance(lines, str) else list(lines), size=size, bold=bold, color=color, align=align)))

    def edge(self, src, dst, ex=1.0, ey=0.5, nx=0.0, ny=0.5, via=(), dashed=False, color="#333333", label="", lpos=0.5):
        self.items.append(("edge", dict(src=src, dst=dst, ex=ex, ey=ey, nx=nx, ny=ny, via=list(via), dashed=dashed, color=color, label=label, lpos=lpos)))

    def point(self, key, px, py):
        n = self.nodes[key]
        return n["x"] + px * n["w"], n["y"] + py * n["h"]


# ====================================================================================== page 1: the main scheme
def main_scheme() -> Page:
    p = Page("Схема поиска", 2100, 1420)
    p.text(20, 8, 1700, 30, "Поиск аналогов: как запрос превращается в список позиций", size=22, bold=True)
    p.text(20, 36, 1700, 20, "Состояние реализованного прототипа на 4 ноября 2026. Пунктир: не реализовано, планируется.", size=13, color="#555555")

    # ------------------------------------------------------------------ bands
    p.band(20, 64, 1760, 310, "#f6f9ff", "#c9d8f5", "ЗАРАНЕЕ: при изменении данных, словаря или кода извлечения (правила пересчёта не требуют)", "#2b4f8f")
    p.band(20, 392, 1760, 140, "#f7fbf8", "#cfe5d6", "ПРИ СТАРТЕ СЕРВИСА: один раз, секунды", "#2f6b46")
    p.band(20, 552, 1760, 830, "#fafafa", "#d6d6d6", "НА КАЖДЫЙ ЗАПРОС: десятки миллисекунд", "#444444")

    # ------------------------------------------------------------------ offline: A1 A2 A3 A5 A4 A6 (the type comes BEFORE the characteristics)
    xs = [40, 330, 620, 910, 1200, 1490]
    y, h, w = 100, 140, 250
    p.node("A1", xs[0], y, w, h, "code", "A1 Загрузка и чистка", ["лист «База позиций», проверка", "структуры; оригинал названия", "и цена не меняются"], "2381 строка")
    p.node("A2", xs[1], y, w, h, "code", "A2 Пометки", ["флаги для всех строк: услуга,", "надбавка, комплект;", "строки не удаляются"], "main 2277 · service 104")
    p.node("A3", xs[2], y, w, h, "code", "A3 Нормализация", ["х x * → x; 1,5 → 1.5; вт = Вт;", "кириллица в кодах;", "«прим.» отдельно"], tags=["словарь"])
    p.node("A5", xs[3], y, w, h, "code", "A5 Тип изделия", ["по словарю: сигнал, не фильтр;", "общий или конкретный;", "опознан у 86 % строк"], tags=["словарь"])
    p.node("A4", xs[4], y, w, h, "code", "A4 Характеристики", ["нужен тип: «7×1.5» у кабеля", "это жилы×сечение;", "порядок сторон сохраняется"], tags=["словарь"])
    p.node("A6", xs[5], y, w, h, "mixed", "A6 Каталог и dense-индекс", ["каталог: код; матрица эмбеддингов", "E5: модель, считается здесь и", "сохраняется; BM25 здесь не строится"])
    for a, b in [("A1", "A2"), ("A2", "A3"), ("A3", "A5"), ("A5", "A4"), ("A4", "A6")]:
        p.edge(a, b)
    p.node("A4p", xs[4], 268, w, 88, "notimpl", "A4+ Остаток через ИИ", ["что код не понял: ИИ пакетно", "[не реализовано]"])
    p.node("APPR", xs[3], 268, w, 88, "humanNI", "Утверждение человеком", ["предложение попадает в каталог", "только после утверждения"])
    p.edge("A4", "A4p", ex=0.5, ey=1, nx=0.5, ny=0, dashed=True)
    p.edge("A4p", "APPR", ex=0, ey=0.5, nx=1, ny=0.5, dashed=True)
    p.edge("APPR", "A6", ex=0.5, ey=1, nx=0.2, ny=1, via=[(xs[3] + 125, 366), (xs[5] + 50, 366)], dashed=True, label="новая версия каталога")

    # ------------------------------------------------------------------ start of the service (right to left)
    p.node("S1", 1490, 428, 250, 90, "code", "Файлы артефактов", ["каталог, индекс, модель,", "правила, словарь"])
    p.node("S2", 1200, 428, 250, 90, "code", "SearchEngine.build", ["проверка манифестов:", "устаревший каталог не стартует"], tags=["правила"])
    p.node("S3", 620, 428, 540, 90, "code", "Каталог и индексы в памяти", ["пулы main 2277 и service 104; BM25 строится по пулам", "E5 только загружается и сверяется; правила и словарь"], shape="store")
    p.edge("A6", "S1", ex=0.8, ey=1, nx=0.8, ny=0, via=[(xs[5] + 200, 428)])
    p.edge("S1", "S2", ex=0, ey=0.5, nx=1, ny=0.5)
    p.edge("S2", "S3", ex=0, ey=0.5, nx=1, ny=0.5)

    # ------------------------------------------------------------------ online row 1 (left to right) and row 2 (right to left)
    y1, y2, h = 600, 880, 140
    p.node("Q", xs[0], y1 + 10, w, 120, "user", "Запрос", ["POST /search", "до 500 символов, limit от 1 до 5"], shape="oval")
    p.node("B1", xs[1], y1, w, h, "code", "B1 Разбор запроса", ["общие функции A3–A5 + режим", "запроса: сдвиг типа, коды, фразы;", "флаги → пул main или service"], tags=["словарь"])
    p.node("B2", xs[2], y1, w, h, "mixed", "B2 Похожие кандидаты", ["BM25: слова и коды, до 100", "E5: смысл текста, до 100", "RRF: слияние, ещё не аналоги"], "выбранный пул → до 50")
    p.node("B3", xs[3], y1, w, h, "code", "B3 Тип: сигнал", ["неизвестный нейтрален;", "общий или иной → проверка;", "конкретный конфликт → B4"], "до 50 → до 50", tags=["правила"])
    p.node("B4", xs[4], y1, w, h, "code", "B4 Техническая проверка", ["предмет, модель/серия и числа;", "по распознанным признакам;", "текстовый score не доказательство"], "до 50", tags=["словарь", "правила"])
    p.node("PG", xs[1], 778, w, 88, "code", "Полнота разбора", ["ограничение запроса не разобрано", "(число с единицей или после", "названия величины) → «проверить»"])
    p.edge("Q", "B1")
    p.edge("B1", "B2")
    p.edge("B2", "B3")
    p.edge("B3", "B4")
    p.edge("B1", "PG", ex=0.5, ey=1, nx=0.5, ny=0)
    p.edge("PG", "B4", ex=1, ey=0.5, nx=0.2, ny=1, via=[(xs[4] + 50, 822)])
    p.edge("S3", "B2", ex=0.2, ey=1, nx=0.5, ny=0, via=[(xs[2] + 125, 518)])
    p.edge("S3", "B3", ex=0.65, ey=1, nx=0.5, ny=0, via=[(xs[3] + 125, 518)])
    p.edge("S3", "B4", ex=0.9, ey=1, nx=0.3, ny=0, via=[(1106, 572), (xs[4] + 75, 572)])

    p.node("B5", xs[4], y2, w, h, "code", "B5 Ранжирование", ["порядок внутри класса:", "0,6 × характеристики +", "0,4 × текст; не вероятность"], "до 50", tags=["правила"])
    p.node("B6", xs[3], y2, w, h, "code", "B6 Списки и статус", ["подтверждённые → found;", "иначе варианты к проверке", "или «не найдено»"], tags=["правила"])
    p.node("UIP", xs[2], y2, w, h, "code", "Основная таблица", ["не больше limit ≤ 5:", "подтверждённые, затем", "«требует проверки»"], tags=["правила"])
    p.node("B7", xs[1], y2, w, h, "code", "B7 Объяснение", ["шаблоном, без ИИ: что совпало,", "что допустимо по правилу,", "чего нет, что не разобрано"])
    p.edge("B4", "B5", ex=0.5, ey=1, nx=0.5, ny=0)
    p.edge("B5", "B6", ex=0, ey=0.5, nx=1, ny=0.5)
    p.edge("B6", "UIP", ex=0, ey=0.5, nx=1, ny=0.5)
    p.edge("UIP", "B7", ex=0, ey=0.5, nx=1, ny=0.5)

    # ------------------------------------------------------------------ result
    p.node("OUT", 40, 1090, 910, 190, "user", "Результат на экране", [
        "1. Есть подтверждённые: «совпадает по указанным параметрам» или «допустимая замена по правилу»;",
        "    «та же позиция по данным названия» только если совпало всё нормализованное название.",
        "2. Подтверждённых нет, есть слабые: «Подтверждённых подходящих позиций не найдено. Ниже варианты для проверки».",
        "3. Ничего подходящего: «Не найдено». Ближайший отказ с причиной: диагностика, не результат.",
        "У каждой позиции: название, цена, оценка и её состав, совпало / отличается / нет данных."], size=13)
    p.edge("B7", "OUT", ex=0.5, ey=1, nx=0.25, ny=0, via=[(xs[1] + 125, 1050), (40 + 227, 1050)])

    # ------------------------------------------------------------------ the future feedback loop (dashed, not implemented)
    p.node("B8", 1000, 1090, 250, 110, "humanNI", "B8 Специалист", ["подтверждает или отклоняет", "пару «запрос → кандидат»", "[не реализовано]"])
    p.node("FB", 1290, 1090, 240, 110, "notimpl", "Журнал подтверждений", ["запрос, кандидат, версия", "каталога, да/нет, причина", "[не реализовано]"], shape="store")
    p.node("EVAL", 1000, 1250, 250, 100, "notimpl", "Контрольные запросы", ["новые пары становятся", "regression-набором", "[не реализовано]"])
    p.node("PROP", 1290, 1250, 240, 100, "notimpl", "Очередь предложений", ["повторы → предложение", "правила или синонима", "[не реализовано]"])
    p.node("APPR2", 1560, 1250, 190, 100, "humanNI", "Утверждает человек", ["новая версия файла", "[не реализовано]"])
    p.edge("OUT", "B8", ex=1, ey=0.5, nx=0, ny=0.5, dashed=True)
    p.edge("B8", "FB", dashed=True)
    p.edge("FB", "B5", ex=0.406, ey=0, nx=0.75, ny=1, dashed=True, label="локальная добавка к порядку", lpos=0.45)
    p.edge("FB", "PROP", ex=0.5, ey=1, nx=0.5, ny=0, dashed=True)
    p.edge("FB", "EVAL", ex=0.2, ey=1, nx=0.5, ny=0, via=[(1338, 1226), (1125, 1226)], dashed=True)
    p.edge("PROP", "APPR2", dashed=True)
    p.edge("APPR2", "RULES", ex=1, ey=0.5, nx=0, ny=0.7, via=[(1795, 1300), (1795, 993)], dashed=True, color="#e08a1e")
    p.edge("APPR2", "DICT", ex=1, ey=0.5, nx=0, ny=0.7, via=[(1795, 1300), (1795, 501)], dashed=True, color="#e08a1e")

    # ------------------------------------------------------------------ the two configuration stores (right column) and the legend
    p.node("DICT", 1830, 410, 250, 130, "human", "Словарь", ["config/dictionaries.yaml", "типы, синонимы, материалы, цвета", "ведёт специалист", "используют: A3 A4 A5 B1 B4"], shape="store")
    p.node("RULES", 1830, 860, 250, 190, "human", "Правила", ["config/rules.yaml", "ведёт специалист, код не меняется", "сечение кабеля строго, камера", "±1 ступень (допущения прототипа)", "используют: B3 B4 B5 B6 таблица"], shape="store")
    p.node("DISTINCTION", 1830, 575, 250, 240, "question", "Два разных решения", [
        "B2: описание похоже —", "предложить кандидата.",
        "B4: требования подтверждены —", "разрешить выдачу как подходящего.",
        "Близость E5 не доказывает", "техническую взаимозаменяемость.",
        "Риски и действия: лист 3."], size=12)
    p.text(1830, 70, 250, 22, "Легенда", size=15, bold=True)
    for i, (cls, lab) in enumerate([("code", "код"), ("model", "модель (ИИ)"), ("mixed", "код и модель"), ("human", "решает человек"), ("notimpl", "не реализовано")]):
        p.node(f"L{i}", 1830, 100 + i * 36, 34, 22, cls, "", [], size=10)
        p.text(1876, 100 + i * 36, 200, 22, lab, size=14)
    p.text(1830, 290, 250, 100, ["Метки «словарь» и «правила»", "у блока: он читает этот файл.", "Числа: сколько позиций остаётся", "после шага (пример из замера)."], size=12, color="#444444")
    return p


# ====================================================================================== page 2: how a pair becomes a verdict
def decision_scheme() -> Page:
    p = Page("Как решается: совпадает, замена, проверить, не подходит", 1900, 1180)
    p.text(20, 8, 1800, 30, "Как система решает по паре «запрос ↔ позиция из базы»", size=22, bold=True)
    p.text(20, 38, 1800, 20, "Это шаг B4 и вердикт B6. Ни один балл на этот результат не влияет: оценка B5 только упорядочивает позиции внутри класса.", size=13, color="#555555")

    cx, w = 560, 460
    p.node("START", cx, 70, w, 85, "user", "Пара: запрос ↔ кандидат из 50", ["после B2 и B3; сравнивается каждая", "характеристика, названная в запросе"], shape="oval")
    p.node("D1", cx, 190, w, 120, "question", "1. Есть конфликт по названной характеристике?", [
        "другое сечение, ток, размер, материал, класс пожарной", "безопасности; другая модель или серия; два разных", "конкретных типа из словаря (розетка ≠ выключатель);",
        "разрешённого правилом отличия недостаточно"], size=13)
    p.node("REJ", 40, 190, 380, 120, "reject", "Не подходит (reject)", ["в выдачу не попадает;", "ближайший по числу конфликтов виден", "как диагностика с причиной"])
    p.node("D2", cx, 360, w, 160, "question", "2. Что-то нельзя проверить?", [
        "у кандидата нет характеристики; порядок сторон не подтверждён;", "значение только в «прим.»; ограничение запроса не разобрано",
        "(число с единицей или после названия величины), если то же", "не записано у позиции (исключение: то же написание, с пометкой);",
        "непроверенный состав; нет слова-предмета; слов мало"], size=13)
    p.node("CHK", 1100, 360, 400, 160, "check", "Проверить (check)", ["«требует проверки»: специалист решает;", "причина названа в карточке;", "в таблице идёт после подтверждённых"])
    p.node("D3", cx, 570, w, 120, "question", "3. Есть достаточное основание подтверждения?", [
        "характеристика или код модели совпали, либо замена", "по правилу (цвет, «уличный», «шт», римский класс", "и один бренд не в счёт); или совпал весь текст;", "или допущение по типу (термометр), с пометкой"], size=13)
    p.node("D4", cx, 740, w, 110, "question", "4. Отличие разрешено правилом?", ["например, камера 3 Мп ↔ 2 Мп при допуске ±1 ступень", "правило названо в карточке (rule_id)"], size=13)
    p.node("SUB", 1100, 740, 400, 110, "substitute", "Допустимая замена (substitute)", ["«допустимо по правилу …», с исходным", "и допустимым значением"])
    p.node("D5", cx, 890, w, 110, "question", "5. Совпало всё основное название?", ["после нормализации чисел, регистра, запятых и пробелов;", "не комплект; синонимы словаря не в счёт"], size=13)
    p.node("SAMEN", 1100, 1030, 400, 110, "same", "Та же позиция по данным названия", ["единственное основание, при котором", "система так пишет"])
    p.node("SAME", 40, 890, 380, 110, "same", "Совпадает по указанным параметрам", ["параметры запроса совпали; остальное не проверялось;", "основание request_constraints; при допущении", "по типу: type_assumption, с пометкой"])
    p.node("NOTE", 40, 1030, 1000, 120, "question", "Чего это решение НЕ доказывает", [
        "• что позиции одна и та же вещь: «труба 25 мм» подойдёт и ПВХ, и ПНД, пока в запросе не назван материал;",
        "• полноты разбора: числовой guard не ловит все словесные требования; нераспознанное свойство может быть потеряно;",
        "• при допущении по типу: диапазон и точность не проверялись, действует допущение автора, не правило заказчика."], size=12)

    p.edge("START", "D1", ex=0.5, ey=1, nx=0.5, ny=0)
    p.edge("D1", "REJ", ex=0, ey=0.5, nx=1, ny=0.5, label="да")
    p.edge("D1", "D2", ex=0.5, ey=1, nx=0.5, ny=0, label="нет")
    p.edge("D2", "CHK", ex=1, ey=0.5, nx=0, ny=0.5, label="да")
    p.edge("D2", "D3", ex=0.5, ey=1, nx=0.5, ny=0, label="нет")
    p.edge("D3", "CHK", ex=1, ey=0.5, nx=0.25, ny=1, via=[(1200, 630)], label="нет: одного типа мало", lpos=0.5)
    p.edge("D3", "D4", ex=0.5, ey=1, nx=0.5, ny=0, label="да")
    p.edge("D4", "SUB", ex=1, ey=0.5, nx=0, ny=0.5, label="да")
    p.edge("D4", "D5", ex=0.5, ey=1, nx=0.5, ny=0, label="нет")
    p.edge("D5", "SAMEN", ex=1, ey=0.5, nx=0, ny=0.5, via=[(1060, 945), (1060, 1085)], label="да")
    p.edge("D5", "SAME", ex=0, ey=0.5, nx=1, ny=0.5, label="нет")

    p.node("WHO", 1560, 80, 320, 260, "human", "Кто что задаёт", [
        "конфликты и допуски:", "config/rules.yaml, специалист;", "что считается подтверждением,", "слабые признаки, лимиты:",
        "policy в том же файле;", "типы, синонимы, материалы:", "config/dictionaries.yaml"], shape="store")
    return p


# ====================================================================================== page 3: text similarity, technical meaning, failure diagnosis
def failure_scheme() -> Page:
    p = Page("Сходство, соответствие и точки отказа", 1820, 1710)
    p.text(30, 8, 1760, 32, "Похожее описание и подходящее оборудование — разные проверки", size=23, bold=True)
    p.text(30, 42, 1760, 24, "B2 предлагает кандидатов; B3/B4 проверяют предмет и требования. Ни BM25, ни E5 не утверждают допустимость замены.", size=14, color="#444444")
    p.band(20, 78, 1780, 290, "#f6f9ff", "#c9d8f5", "ДВА СПОСОБА НАЙТИ → ОДИН МЕХАНИЗМ ТЕХНИЧЕСКОГО РЕШЕНИЯ", "#2b4f8f")
    p.node("FQ", 40, 180, 230, 110, "user", "Запрос и его разбор", ["какой предмет нужен;", "что обязательно"], shape="oval", size=14)
    p.node("LEX", 340, 118, 380, 100, "code", "BM25: слова, коды, числа", ["Риск: параметры совпали,", "а предметы разные."], size=14)
    p.node("SEM", 340, 250, 380, 100, "model", "E5: близость смысла текста", ["Перефразировки и общая тема.", "Не доказывает совместимость."], size=14)
    p.node("MERGE", 800, 176, 350, 140, "code", "RRF: объединить предложения", ["до 100 + до 100 → до 50;", "это ещё только кандидаты;", "потерянного здесь B4 не вернёт"], size=14)
    p.node("TECH", 1230, 176, 550, 140, "code", "B3/B4: соответствие по данным названия", [
        "Предмет, модель/серия, распознанные признаки.",
        "Совпало / разрешено правилом / конфликт / неизвестно.",
        "За пределами названия нужны сведения специалиста."], size=14)
    p.edge("FQ", "LEX", via=[(305, 235), (305, 168)])
    p.edge("FQ", "SEM", via=[(305, 235), (305, 300)])
    p.edge("LEX", "MERGE", nx=0, ny=0.25, via=[(760, 168), (760, 211)])
    p.edge("SEM", "MERGE", nx=0, ny=0.75, via=[(760, 300), (760, 281)])
    p.edge("MERGE", "TECH")
    p.text(40, 381, 1740, 26, "Где логика может ломаться и что проверять: правый столбец — действия при диагностике, а не обещание готовности всех профилей.", size=14, color="#444444")

    # Each row is a diagnostic route, not an extra runtime branch.
    columns = [(40, 240, "Этап"), (320, 440, "Ошибка / признак"), (800, 600, "Как разбирать и исправлять"), (1440, 340, "Отдельная проверка")]
    for x, w, title in columns:
        p.text(x, 421, w, 24, title, size=16, bold=True)
    rows = [
        ("PARSE", "A3–A5 / B1: разбор", ["Не распознано требование:", "число, единица, слово, роль."],
         "Из запроса исчезло ограничение", ["100 МЛ, 16 портов, 1-кл.;", "сталь трубы принята за свойство прибора;", "потеряно словесное требование;", "число в обозначении (KLS 300) не размер."],
         ["Сравнить исходный текст, атрибуты и остаток;", "нормализовать единицы с правильным масштабом;", "задать профиль: предмет, назначение, роль признака.", "Сохранить evidence и словесные ограничения;", "неясное обязательное требование → check."],
         ["Тесты эквивалентных записей;", "известный конфликт и no-data;", "проверка роли признака."]),
        ("RETRIEVE", "B2: набор кандидатов", ["Эталон не попал в top-50.", "B4 его уже не оценит."],
         "Не найдено или потеряно в B2", ["Другие слова, код, неверный пул;", "BM25 и E5 могут потерять нужную строку.", "Пустой экран не доказывает сбой B2."],
         ["По trace проверить пул и ID кандидатов;", "сопоставить BM25 и гибрид на одном наборе;", "исправлять токены/поиск только после атрибуции.", "Не отрезать допустимые замены и unknown-факты."],
         ["candidate_hit@50;", "candidate_recall@50;", "ранг эталона до matching."]),
        ("MATCH", "B3/B4: сравнение", ["Похожие параметры", "скрывают другой товар."],
         "Опасное ложное подтверждение", ["Розетка вместо выключателя;", "труба вместо толщиномера;", "две клавиши или другая модель."],
         ["Проверить предмет и каждое обязательное требование;", "известное несовпадение → reject, не компенсация score;", "неизвестное → check; замена только с правилом.", "Сходство текста не является допуском специалиста."],
         ["Фиксированные пары без B2;", "все ложные подтверждения;", "разрезы: предмет / требования."]),
        ("UNCERTAINTY", "B4/B6: уверенность", ["Все сведения есть, но check;", "конфликт назван неизвестным."],
         "Неверная оценка неопределённости", ["Пример первого прогона: P10 → check;", "иная серия → не просто «нет размера»;", "состав комплекта не сверён."],
         ["Разделить match, conflict и unknown по признакам;", "извлечь качественные свойства и требуемый состав;", "если данных действительно нет — запросить уточнение.", "Подтверждение по типу не снимает остальных требований."],
         ["Матрица решений человека;", "лишний check для допустимой пары;", "check при известном конфликте."]),
        ("DISPLAY", "B5–B7: отбор и вывод", ["Кандидат проверен,", "но нужная строка не показана."],
         "Выдача скрыла результат или риск", ["Ранг/лимит, слабое evidence, фильтр таблицы;", "ошибочная строка ниже верного top-1;", "review смешан с подтверждённым found."],
         ["Проследить класс → вердикт → ранг → on_screen;", "считать results и table раздельно, проверять все строки;", "показывать причину, исходное название и цену.", "Не выдавать высокий балл за вероятность пригодности."],
         ["Confirmed / Display Hit@5;", "состояния: empty / review / found;", "число ошибок, не одна средняя."]),
    ]
    for i, (key, stage, stage_lines, problem, problem_lines, action, metric) in enumerate(rows):
        y = 456 + i * 166
        p.node(key, 40, y, 240, 146, "question", stage, stage_lines, size=14)
        p.node(key + "_RISK", 320, y, 440, 146, "reject", problem, problem_lines, size=14)
        p.node(key + "_ACTION", 800, y, 600, 146, "human", "Действие при ошибке", action, size=14)
        p.node(key + "_CHECK", 1440, y, 340, 146, "code", "Как увидеть результат", metric, size=14)
        p.edge(key, key + "_RISK")
        p.edge(key + "_RISK", key + "_ACTION")
        p.edge(key + "_ACTION", key + "_CHECK")

    p.node("TEXT_EXAMPLE", 40, 1314, 550, 180, "question", "Похожи словами, но разные товары", [
        "Выключатель 250 В / 16 А ↔ розетка 250 В / 16 А.",
        "Известный конфликт предмета → reject.",
        "Толщиномер стальных труб ↔ стальная труба:",
        "материал объекта измерения не делает трубу прибором.",
        "Тип не установлен → не подтверждать по общим числам."], size=14)
    p.node("MEANING_EXAMPLE", 635, 1314, 550, 180, "question", "Разные слова, одно назначение", [
        "Пирометр ↔ инфракрасный термометр.",
        "Словарь или E5 помогает предложить кандидата.",
        "Затем проверяются заданные бренд и параметры;",
        "допустимость замены всё равно требует основания."], size=14)
    p.node("FORMAT_EXAMPLE", 1230, 1314, 550, 180, "question", "Оформление или смысл требования?", [
        "Песок II класса: запятые не меняют товар.",
        "L 3000 mm = L=3000 мм: это одна длина.",
        "Сталь 1,5 мм без слова «толщина»: нужен профиль",
        "или уточнение, а не молчаливое предположение."], size=14)
    p.node("SNAPSHOT", 40, 1530, 1740, 145, "code", "Срез контрольной проверки · 4 ноября 2026", [
        "Контрольный набор из 20 запросов (12 с ответом, 8 без), разобранный и исправленный: все 12 эталонов среди 50 кандидатов, подтверждено 12/12, ложных строк 0, отказов 8/8.",
        "Первый прогон до исправлений дал 5 из 12 при 8 ложных подтверждениях: цифры после исправлений это регрессия, а не независимая оценка.",
        "Общая полнота понимания не доказана; независимая проверка описана в eval/PROTOCOL.md; границы реализации — в README и ARCHITECTURE."], size=14)
    return p


# ====================================================================================== renderers
def _label_lines(n: dict) -> list[tuple[str, str]]:
    out = []
    if n["title"]:
        out.append(("b", n["title"]))
    out += [("n", t) for t in n["lines"]]
    if n["count"]:
        out.append(("b", "▸ " + n["count"]))
    return out


def _edge_points(p: Page, e: dict) -> list[tuple[float, float]]:
    s, t = p.point(e["src"], e["ex"], e["ey"]), p.point(e["dst"], e["nx"], e["ny"])
    if e["via"]:
        return [s, *e["via"], t]
    if abs(s[0] - t[0]) < 1 or abs(s[1] - t[1]) < 1:
        return [s, t]
    if e["ex"] in (0.0, 1.0):                                  # leaves sideways: a Z with a vertical middle
        mx = (s[0] + t[0]) / 2
        return [s, (mx, s[1]), (mx, t[1]), t]
    my = (s[1] + t[1]) / 2
    return [s, (s[0], my), (t[0], my), t]


def render_svg(p: Page) -> str:
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {p.width} {p.height}" width="{p.width}" height="{p.height}" font-family="Helvetica, Arial, sans-serif">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>'
         '<linearGradient id="mix" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#e8f1ff"/><stop offset="1" stop-color="#e3f5e8"/></linearGradient></defs>',
         f'<rect width="{p.width}" height="{p.height}" fill="#ffffff"/>']
    for kind, d in p.items:
        if kind == "band":
            o.append(f'<rect x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" rx="14" fill="{d["fill"]}" stroke="{d["stroke"]}" stroke-width="1.5"/>')
            o.append(f'<text x="{d["x"] + 18}" y="{d["y"] + 24}" font-size="15" font-weight="700" fill="{d["font"]}">{escape(d["title"])}</text>')
        elif kind == "text":
            anchor = {"left": "start", "center": "middle", "right": "end"}[d["align"]]
            tx = d["x"] if d["align"] == "left" else d["x"] + d["w"] / 2
            for i, ln in enumerate(d["lines"]):
                o.append(f'<text x="{tx}" y="{d["y"] + d["size"] + i * (d["size"] + 4)}" font-size="{d["size"]}" text-anchor="{anchor}"'
                         f'{" font-weight=\"700\"" if d["bold"] else ""} fill="{d["color"]}">{escape(ln)}</text>')
        elif kind == "node":
            o.append(_svg_node(d))
    for kind, d in p.items:                                    # edges after nodes: arrowheads stay visible
        if kind == "edge":
            pts = _edge_points(p, d)
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            dash = ' stroke-dasharray="7 5"' if d["dashed"] else ""
            o.append(f'<polyline points="{path}" fill="none" stroke="{d["color"]}" stroke-width="2"{dash} marker-end="url(#arr)"/>')
            if d["label"]:
                # label on the longest segment
                seg = max(range(len(pts) - 1), key=lambda i: abs(pts[i + 1][0] - pts[i][0]) + abs(pts[i + 1][1] - pts[i][1]))
                (x1, y1), (x2, y2) = pts[seg], pts[seg + 1]
                lx, ly = x1 + (x2 - x1) * d["lpos"], y1 + (y2 - y1) * d["lpos"]
                wlab = 7.2 * len(d["label"]) + 10
                o.append(f'<rect x="{lx - wlab / 2:.1f}" y="{ly - 10:.1f}" width="{wlab:.1f}" height="20" rx="4" fill="#ffffff" opacity="0.92"/>')
                o.append(f'<text x="{lx:.1f}" y="{ly + 5:.1f}" font-size="12.5" text-anchor="middle" fill="#333333">{escape(d["label"])}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def _svg_node(n: dict) -> str:
    fill, stroke, font, dashed = COLORS[n["cls"]]
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    fill_attr = "url(#mix)" if "|" in fill else fill
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    if n["shape"] == "oval":
        body = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(h / 2, 50)}" fill="{fill_attr}" stroke="{stroke}" stroke-width="2"{dash}/>'
    elif n["shape"] == "store":
        body = (f'<path d="M{x},{y + 14} q0,-14 {w / 2},-14 q{w / 2},0 {w / 2},14 v{h - 28} q0,14 -{w / 2},14 q-{w / 2},0 -{w / 2},-14 z" fill="{fill_attr}" stroke="{stroke}" stroke-width="2"{dash}/>'
                f'<path d="M{x},{y + 14} q0,14 {w / 2},14 q{w / 2},0 {w / 2},-14" fill="none" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    else:
        body = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill_attr}" stroke="{stroke}" stroke-width="2"{dash}/>'
    lines = _label_lines(n)
    out = [body]
    if lines:
        size = n["size"]
        lh = size + 5
        top = y + (h - len(lines) * lh) / 2 + (22 if n["shape"] == "store" else 0) / 2
        for i, (kind, t) in enumerate(lines):
            out.append(f'<text x="{x + w / 2}" y="{top + (i + 1) * lh - 5:.1f}" font-size="{size + (1 if kind == "b" and i == 0 else 0)}" text-anchor="middle"'
                       f'{" font-weight=\"700\"" if kind == "b" else ""} fill="{font}">{escape(t)}</text>')
    for i, tag in enumerate(n["tags"]):
        tw = 62
        tx = x + w - 6 - (len(n["tags"]) - i) * (tw + 4) + 4
        out.append(f'<rect x="{tx}" y="{y - 9}" width="{tw}" height="18" rx="9" fill="#ffffff" stroke="{TAG[tag]}" stroke-width="1.5"/>'
                   f'<text x="{tx + tw / 2}" y="{y + 4}" font-size="11.5" text-anchor="middle" font-weight="700" fill="{TAG[tag]}">{tag}</text>')
    return "\n".join(out)


def render_drawio_page(p: Page, pid: str) -> str:
    cells: list[str] = []
    counter = [10]

    def nid():
        counter[0] += 1
        return f"{pid}e{counter[0]}"

    def html(n):
        return "<br>".join(("<b>" + escape(t) + "</b>") if k == "b" else escape(t) for k, t in _label_lines(n))
    for kind, d in p.items:
        if kind == "band":
            style = (f"rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor={d['fill']};strokeColor={d['stroke']};strokeWidth=1.5;verticalAlign=top;"
                     f"align=left;spacingLeft=14;spacingTop=6;fontStyle=1;fontSize=14;fontColor={d['font']};")
            cells.append(f'<mxCell id="{nid()}" value={quoteattr(escape(d["title"]))} style="{style}" vertex="1" parent="1"><mxGeometry x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" as="geometry"/></mxCell>')
        elif kind == "text":
            style = f"text;html=1;whiteSpace=wrap;verticalAlign=top;align={d['align']};fontSize={d['size']};fontColor={d['color']};fontStyle={1 if d['bold'] else 0};"
            value = "<br>".join(escape(x) for x in d["lines"])
            cells.append(f'<mxCell id="{nid()}" value={quoteattr(value)} style="{style}" vertex="1" parent="1"><mxGeometry x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" as="geometry"/></mxCell>')
        elif kind == "node":
            fill, stroke, font, dashed = COLORS[d["cls"]]
            grad = ""
            if "|" in fill:
                fill, g2 = fill.split("|")
                grad = f"gradientColor={g2};gradientDirection=east;"
            base = f"whiteSpace=wrap;html=1;fillColor={fill};{grad}strokeColor={stroke};fontColor={font};fontSize={d['size']};strokeWidth=2;" + ("dashed=1;" if dashed else "")
            if d["shape"] == "oval":
                base = "rounded=1;arcSize=40;" + base
            elif d["shape"] == "store":
                base = "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=14;" + base
            else:
                base = "rounded=1;arcSize=8;" + base
            cells.append(f'<mxCell id="{pid}_{d["key"]}" value={quoteattr(html(d))} style="{base}" vertex="1" parent="1"><mxGeometry x="{d["x"]}" y="{d["y"]}" width="{d["w"]}" height="{d["h"]}" as="geometry"/></mxCell>')
            for i, tag in enumerate(d["tags"]):
                tx = d["x"] + d["w"] - 6 - (len(d["tags"]) - i) * 66 + 4
                st = f"rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={TAG[tag]};strokeWidth=1.5;fontColor={TAG[tag]};fontStyle=1;fontSize=11;"
                cells.append(f'<mxCell id="{nid()}" value={quoteattr(tag)} style="{st}" vertex="1" parent="1"><mxGeometry x="{tx}" y="{d["y"] - 9}" width="62" height="18" as="geometry"/></mxCell>')
    for kind, d in p.items:
        if kind == "edge":
            style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeWidth=2;strokeColor={d['color']};"
                     f"exitX={d['ex']};exitY={d['ey']};exitDx=0;exitDy=0;entryX={d['nx']};entryY={d['ny']};entryDx=0;entryDy=0;fontSize=12;labelBackgroundColor=#ffffff;"
                     + ("dashed=1;" if d["dashed"] else ""))
            pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in d["via"])
            geo = f'<mxGeometry relative="1" as="geometry">' + (f'<Array as="points">{pts}</Array>' if pts else "") + "</mxGeometry>"
            cells.append(f'<mxCell id="{nid()}" value={quoteattr(escape(d["label"]))} style="{style}" edge="1" parent="1" source="{pid}_{d["src"]}" target="{pid}_{d["dst"]}">{geo}</mxCell>')
    return (f'<diagram name={quoteattr(p.name)} id="{pid}"><mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
            f'page="1" pageScale="1" pageWidth="{p.width}" pageHeight="{p.height}" math="0" shadow="0"><root><mxCell id="{pid}0"/><mxCell id="1" parent="{pid}0"/>'
            + "".join(cells).replace('parent="1"', f'parent="1"') + "</root></mxGraphModel></diagram>")


def main() -> None:
    pages = [(main_scheme(), "scheme.svg"), (decision_scheme(), "scheme_decision.svg"), (failure_scheme(), "scheme_failures.svg")]
    targets = [DOCS / "scheme.drawio", *(DOCS / name for _, name in pages)]
    if "--overwrite" not in sys.argv and any(t.exists() for t in targets):
        sys.exit("Файлы схемы уже существуют. Источник: scripts/build_scheme.py; генерация заменяет draw.io и SVG.\n"
                 "Для пересоздания из описания: python3 -I scripts/build_scheme.py --overwrite")
    for page, name in pages:
        (DOCS / name).write_text(render_svg(page), encoding="utf-8")
    xml = '<mxfile host="app.diagrams.net">' + "".join(render_drawio_page(page, f"p{i}") for i, (page, _) in enumerate(pages, 1)) + "</mxfile>"
    (DOCS / "scheme.drawio").write_text(xml, encoding="utf-8")
    print("written", *(t.name for t in targets))


if __name__ == "__main__":
    main()
