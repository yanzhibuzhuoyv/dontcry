"""Stress test: 100 synthetic documents — ingest, query, verify, cleanup."""

import os, sys, tempfile, shutil, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
os.environ["RAG_EMBEDDING_PROVIDER"] = "local"
os.environ["RAG_EMBEDDING_MODEL"] = "BAAI/bge-small-zh-v1.5"
os.environ["HF_HUB_OFFLINE"] = "1"

PASS = FAIL = 0
DOC_COUNT = 100


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


TOPICS = [
    ("机器学习", "监督学习通过标注数据训练模型，梯度下降是最常用的优化算法。深度学习使用多层神经网络。"
              "Transformer架构自2017年提出以来成为NLP领域主流。BERT和GPT代表了编码器和解码器的设计思路。"),
    ("物理学", "牛顿三大定律描述宏观物体运动规律。相对论揭示时空与物质的关系。量子力学解释微观粒子波粒二象性。"
              "能量守恒是物理学最基本的原理。熵增原理表明孤立系统熵永不减少。"),
    ("中国历史", "唐朝是中国历史上最繁荣的朝代，长安是当时世界最大城市。宋朝发明了活字印刷术。明朝郑和七下西洋。"
              "清朝康乾盛世持续百年繁荣。鸦片战争标志着中国近代史的开端。"),
    ("Python编程", "Python是解释型高级语言，以简洁易读著称。装饰器可在不修改函数的情况下增加功能。"
              "上下文管理器通过with语句自动管理资源。生成器使用yield关键字实现惰性求值。"),
    ("音乐理论", "十二平均律将八度音程等分为十二个半音。和弦由三个以上不同音高的音组成。"
              "奏鸣曲式是古典音乐最常用的曲式。赋格是一种复调音乐形式。"),
    ("生物学", "DNA双螺旋结构由沃森和克里克于1953年发现。细胞是生命的基本单位。线粒体是细胞的能量工厂。"
              "自然选择是达尔文进化论的核心。基因表达调控决定细胞分化方向。"),
    ("经济学", "供需关系决定市场价格。GDP衡量经济总量。通货膨胀导致货币购买力下降。"
              "凯恩斯主义主张通过政府支出调节经济。比较优势理论解释国际贸易基础。"),
    ("天文学", "太阳系有八大行星。黑洞引力极强连光都无法逃逸。银河系是棒旋星系。"
              "哈勃发现宇宙在膨胀。暗物质和暗能量构成宇宙大部分质量。" ),
    ("心理学", "弗洛伊德提出潜意识理论。巴甫洛夫条件反射奠定了行为主义基础。皮亚杰认知发展理论将儿童思维分四个阶段。"
              "马斯洛需求层次理论从低到高排列人类需求。认知失调解释态度与行为不一致时的心理状态。"),
    ("烹饪", "川菜以麻辣著称宫保鸡丁是代表作。粤菜追求原汁原味白切鸡考验火候。鲁菜讲究刀工调味。"
              "苏菜精致细腻松鼠鳜鱼声名远扬。发酵是制作酱油和醋的关键工艺。"),
    ("建筑学", "哥特式建筑以尖拱和飞扶壁为特征。文艺复兴建筑回归古典比例。现代主义强调功能决定形式。"
              "巴洛克风格追求华丽动感。斗拱是中国传统建筑的独特构件。"),
    ("电影艺术", "蒙太奇是电影剪辑基本技法。长镜头保持时空完整性。新浪潮运动挑战传统叙事方式。"
              "色彩影响观众情绪感知。声音设计营造沉浸式体验。斯坦尼斯拉夫斯基体系奠定方法派表演基础。"),
    ("数学", "微积分由牛顿和莱布尼茨分别创立。欧几里得几何建立公理化体系。哥德尔不完备定理揭示形式系统局限性。"
              "概率论研究随机现象规律。数论探索整数深层性质。拓扑学关心连通性而非尺寸。"),
    ("体育", "足球是世界最受欢迎的运动。马拉松源于希波战争。奥林匹克精神强调参与和公正。"
              "间歇性训练提高心肺功能。拉伸有助于预防运动损伤。"),
    ("哲学", "苏格拉底用提问法探寻真理。柏拉图提出理念论。亚里士多德奠定逻辑学基础。笛卡尔我思故我在开启近代哲学。"
              "康德区分现象界与物自体。存在主义强调个体自由与选择。"),
]

print(f"\n=== Step 1: Generating {DOC_COUNT} documents ===")
td = tempfile.mkdtemp()
docs_dir = Path(td) / "docs"
docs_dir.mkdir()

for i in range(DOC_COUNT):
    idx = i % len(TOPICS)
    topic, content = TOPICS[idx]
    text = f"# {topic} — 文档{i+1:03d}\n\n文档编号 {i+1:03d}\n\n{content}"
    (docs_dir / f"doc_{i+1:03d}.txt").write_text(text, encoding="utf-8")

n_generated = len(list(docs_dir.glob("*.txt")))
check(n_generated == DOC_COUNT, f"generated {n_generated}/{DOC_COUNT} documents")

print(f"\n=== Step 2: Ingesting {DOC_COUNT} documents ===")
os.environ["RAG_VECTOR_STORE_DIR"] = str(Path(td) / "rag_index")

from rag_system.rag import RAGSystem
from rag_system.config import load_rag_config

rag = RAGSystem(load_rag_config())
r = rag.ingest(str(docs_dir))

check(r["files"] >= 90, f"ingested {r['files']} files")
check(r["chunks"] >= 90, f"chunks: {r['chunks']}")
check(r["skipped"] == 0, f"first ingest: 0 skipped")

print(f"\n=== Step 3: Retrieval accuracy ===")
from rag_system.vector_store import VectorStore
store = VectorStore.load(str(Path(td) / "rag_index"))
check(store.count >= 90, f"index has {store.count} vectors")

test_queries = [
    ("梯度下降优化算法", "机器学习"),
    ("DNA双螺旋结构", "生物学"),
    ("哥特式建筑飞扶壁", "建筑学"),
    ("弗洛伊德潜意识理论", "心理学"),
    ("唐朝长安丝绸之路", "中国历史"),
    ("十二平均律半音", "音乐理论"),
    ("微积分牛顿莱布尼茨", "数学"),
    ("供需关系市场价格", "经济学"),
    ("马拉松希波战争", "体育"),
    ("笛卡尔我思故我在", "哲学"),
    ("量子力学波粒二象性", "物理学"),
    ("蒙太奇剪辑技法", "电影艺术"),
    ("川菜麻辣宫保鸡丁", "烹饪"),
    ("太阳系八大行星", "天文学"),
    ("装饰器上下文管理器", "Python编程"),
]

qp = 0
for query, expected in test_queries:
    results = store.search(rag._embedder.embed_query(query), k=3)
    if not results:
        continue
    top_sources = [Path(r.source).stem for r in results[:3] if r.score > 0.35]
    # Find doc index to check topic
    matched = False
    for src in top_sources:
        m = re.search(r'(\d+)', src)
        if m:
            didx = int(m.group(1)) - 1
            if 0 <= didx < DOC_COUNT and TOPICS[didx % len(TOPICS)][0] == expected:
                matched = True
                break
    if matched:
        qp += 1
    else:
        top_topics = set()
        for src in top_sources:
            m = re.search(r'(\d+)', src)
            if m:
                didx = int(m.group(1)) - 1
                if 0 <= didx < DOC_COUNT:
                    top_topics.add(TOPICS[didx % len(TOPICS)][0])
        print(f"  [WARN] '{query}': expected '{expected}', got {top_topics}")

check(qp >= 12, f"retrieval accuracy: {qp}/{len(test_queries)}")

print(f"\n=== Step 4: Incremental update ===")
r2 = rag.ingest(str(docs_dir))
check(r2["skipped"] >= 95, f"re-ingest skips {r2['skipped']} unchanged docs")

for i in [5, 33, 67]:
    path = docs_dir / f"doc_{i+1:03d}.txt"
    path.write_text(f"# 修改后文档{i+1}\n新增关键词：量子纠缠光子实验{i*73}。", encoding="utf-8")

r3 = rag.ingest(str(docs_dir), force=True)
check(r3.get("updated", 0) == 3, f"detected 3 modified docs (got {r3.get('updated',0)})")

print(f"\n=== Step 5: Memory ===")
from rag_system.session_memory import SessionMemory
mem = SessionMemory(base_dir=Path(td) / "mem")
rm = mem.end_session(
    f"完成{DOC_COUNT}文档压力测试。检索准确率{qp}/{len(test_queries)}，增量更新检测3个修改文档。",
    slug="stress-100", session_date="2026-07-03"
)
check(len(rm["prompts"]) > 0, f"memory: {len(rm['prompts'])} prompts")

print(f"\n=== Step 6: Cleanup ===")
shutil.rmtree(td, ignore_errors=True)
check(not Path(td).exists(), "all test data cleaned")

print(f"\n{'='*60}")
print(f"  Stress Test: {DOC_COUNT} docs, {PASS + FAIL} tests")
print(f"  Passed: {PASS}  Failed: {FAIL}")
print(f"{'='*60}")

if FAIL > 0:
    sys.exit(1)
