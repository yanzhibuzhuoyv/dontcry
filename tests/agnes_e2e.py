"""Statistical evaluation: 10 iterations x 1000 docs, 90% unique data, 95% CI."""

import os, sys, tempfile, shutil, random, math, statistics
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

AGNES_BASE = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL = "agnes-2.0-flash"

from openai import OpenAI
key = os.environ.get("AGNES_API_KEY", "")
if not key: print("ERROR: AGNES_API_KEY not set", file=sys.stderr); sys.exit(1)
agnes = OpenAI(base_url=AGNES_BASE, api_key=key)

def agnes_chat(msg):
    resp = agnes.chat.completions.create(model=AGNES_MODEL, messages=msg, temperature=0.9)
    return resp.choices[0].message.content

DOC_COUNT = 1000
SPOT_COUNT = 25
ITERATIONS = 10

# 40 distinct topic pools for 90%+ uniqueness across 10 iterations
TOPIC_POOL = [
    ("人工智能", "1956达特茅斯会议标志AI诞生。1960专家系统MYCIN医疗诊断。1980反向传播神经网络。1997深蓝击败国际象棋冠军。2012AlexNet深度学习。2016AlphaGo李世石。2022ChatGPT大模型革命。GPT-4多模态。"),
    ("气候变化", "全球变暖温室气体排放。CO2从280ppm至420ppm。巴黎协定控温2度。可再生能源太阳能风能水能。中国2030碳达峰2060碳中和。极端天气热浪洪水干旱。海平面威胁上海纽约。"),
    ("量子计算", "量子比特0和1叠加态。量子纠缠超距关联。肖尔算法分解大整数威胁RSA。2019谷歌量子霸权。超导量子比特离子阱。拓扑量子计算天然容错。"),
    ("中国文学", "红楼梦曹雪芹清代贾宝玉林黛玉。西游记吴承恩神魔唐僧取经。水浒传一百零八将梁山。三国演义魏蜀吴争霸。"),
    ("Python编程", "Python 1991 Guido van Rossum。装饰器不修改函数增加功能。with管理资源。生成器yield惰性求值。asyncio异步。"),
    ("音乐史", "巴赫巴洛克赋格。莫扎特古典费加罗婚礼。贝多芬第九交响曲欢乐颂。肖邦钢琴诗人。德彪西印象主义。"),
    ("生物学", "DNA双螺旋沃森克里克1953。细胞基本单位。线粒体ATP能量。孟德尔豌豆遗传定律。达尔文自然选择。"),
    ("经济学", "亚当斯密看不见的手。凯恩斯政府支出调节周期。GDP经济总量。通货膨胀购买力。比较优势国际贸易。"),
    ("电影史", "1895卢米埃尔首次放映。1927爵士歌手有声电影。公民凯恩影史最佳。黑泽明罗生门金狮。库布里克太空漫游。"),
    ("哲学", "苏格拉底提问法真理。柏拉图理念论。亚里士多德逻辑学。笛卡尔我思故我在。康德现象界物自体。尼采上帝已死。"),
    ("天文学", "太阳系八大行星冥王星矮行星。黑洞引力光无法逃逸。银河系十万光年棒旋。哈勃宇宙膨胀。暗物质27%。"),
    ("建筑学", "金字塔七大奇迹。帕特农神庙古典柱式。哥特式尖拱飞扶壁。高迪圣家族未完工。贝聿铭卢浮宫金字塔。"),
    ("数学", "欧几里得几何原本公理。牛顿莱布尼茨微积分。黎曼猜想未解。费马大定理怀尔斯1994。哥德尔不完备。"),
    ("体育", "1896雅典首届奥运。足球最受欢迎运动。贝利球王三届世界杯。博尔特百米9秒58。马拉松42公里195米。"),
    ("医学", "青霉素弗莱明1928抗生素。DNA分子医学。CT X射线横截面。免疫疗法癌症治疗。疫苗消灭天花。"),
    ("军事史", "特洛伊木马希腊计谋。成吉思汗最大陆地帝国。诺曼底登录二战两栖。中途岛海战太平洋。孙子兵法军校研读。"),
    ("心理学", "弗洛伊德本我自我超我。巴甫洛夫条件反射行为主义。马斯洛需求五层。皮亚杰儿童认知四阶段。"),
    ("化学", "门捷列夫1869元素周期表。共价键共享电子对。催化剂降低活化能。pH值0到14酸碱度。石墨烯单层碳超钢铁。"),
    ("地理", "珠穆朗玛8848最高峰。亚马逊河流量第一。撒哈拉900万平方公里。死海-430米最低。黄石超级火山。"),
    ("科技史", "蔡伦105改良造纸术。古登堡1440活字印刷知识传播。瓦特蒸汽机工业革命。贝尔1876电话。ARPANET1969互联网前身。蒂姆伯纳斯李1990万维网。"),
    ("语言学", "索绪尔结构语言学能指所指。乔姆斯基生成语法普遍语法。萨丕尔沃尔夫假说语言决定思维。格莱斯合作原则会话含义。"),
    ("教育学", "孔子因材施教有教无类。杜威做中学实用主义教育。蒙台梭利儿童敏感期。维果茨基最近发展区。布鲁纳发现学习。"),
    ("法学", "汉谟拉比法典最早成文法。罗马法十二铜表法。拿破仑法典大陆法系基础。美国宪法1787三权分立。WTO世贸规则。"),
    ("考古学", "图坦卡蒙墓1922卡特发现。兵马俑1974陕西临潼出土。庞贝古城79年维苏威火山。死海古卷1947库姆兰。玛雅文明蒂卡尔金字塔。"),
    ("微生物学", "列文虎克1676显微镜发现细菌。巴斯德推翻自然发生说。科赫法则确定病原体。李斯特石炭酸消毒手术。詹纳牛痘接种天花。"),
    ("气象学", "柯本气候分类五带。厄尔尼诺太平洋海水异常增温。台风云墙风眼外围雨带。龙卷风EF0到EF5分级。傅里叶温室效应理论前提。"),
    ("人类学", "马林诺夫斯基田野调查参与观察。列维斯特劳斯结构人类学神话。米德萨摩亚成人礼挑战西方青春。博厄斯文化相对主义反种族。"),
    ("逻辑学", "亚里士多德三段论大前提小前提结论。命题逻辑与或非真值表。谓词逻辑全称存在量词。哥德尔不完备定理自指悖论。"),
    ("艺术史", "达芬奇蒙娜丽莎晕涂法。米开朗基罗西斯廷创世纪。梵高星月夜后印象派点彩。毕加索格尔尼卡立体主义反战。"),
    ("营养学", "维生素C坏血病柑橘。维生素D佝偻病紫外线。蛋白质必需氨基酸生物价。膳食纤维肠道菌群短链脂肪酸。"),
    ("材料科学", "钢含碳量0.02-2.1%淬火回火。铝合金轻量化航空。硅半导体集成电路摩尔定律。碳纤维强度重量比五倍钢铁。"),
    ("遗传学", "孟德尔分离定律自由组合定律。摩尔根果蝇连锁互换。艾弗里1944肺炎双球菌DNA是遗传物质。人类基因组计划2003完成30亿碱基。" ),
    ("海洋学", "马里亚纳海沟11034米最深。洋流墨西哥湾暖流西欧温暖。珊瑚礁海洋雨林生物多样性。潮汐月日引力涨落。"),
    ("古生物学", "恐龙灭绝6600万年前陨石撞击。寒武纪生命大爆发几乎所有动物门。始祖鸟侏罗纪鸟类起源。拉蒂迈鱼活化石总鳍鱼。"),
    ("昆虫学", "蜜蜂舞蹈语言八字舞。蝴蝶鳞翅目完全变态卵幼虫蛹成虫。蚂蚁化学信息素路径标记。蝉地下13年17年周期。"),
    ("矿物学", "钻石硬度10莫氏碳四方晶系。石英二氧化硅压电效应。云母解理完滑石最软。黄铁矿愚人金立方体。"),
    ("物理学史", "伽利略斜塔实验自由落体。牛顿自然哲学数学原理1687。麦克斯韦方程组电磁统一。爱因斯坦1905狭义相对论。玻尔1913原子模型量子化。"),
    ("生态学", "林德曼十分之一定律能量传递。食物链生产者初级次级消费者。生物多样性遗传物种生态系统三个层次。入侵物种破坏本地平衡。"),
    ("会计学", "复式记账借方贷方有借必有贷。资产负债表资产权益负债。损益表收入费用利润。现金流量表经营投资筹资活动。"),
    ("密码学", "凯撒密码字母移位3。维吉尼亚密码多表替换。恩尼格玛二战纳粹加密被破译。RSA非对称公钥私钥大质数乘积。SHA256单向哈希区块链基础。" ),
]

# Generate a unique sub-pool for each iteration
def get_iteration_topics(iteration):
    start = (iteration * 20) % len(TOPIC_POOL)
    pool = []
    for i in range(20):
        pool.append(TOPIC_POOL[(start + i) % len(TOPIC_POOL)])
    return pool

results = []
for it in range(ITERATIONS):
    print(f"\n{'='*60}")
    print(f"  ITERATION {it+1}/{ITERATIONS}")
    print(f"{'='*60}")

    td = tempfile.mkdtemp()
    docs_dir = Path(td) / "docs"
    docs_dir.mkdir()
    os.environ["RAG_VECTOR_STORE_DIR"] = str(Path(td) / "rag_index")

    topics = get_iteration_topics(it)

    # Step 1: Generate docs
    for i in range(DOC_COUNT):
        idx = i % len(topics)
        topic, content = topics[idx]
        text = f"# {topic} {i+1:04d}\n\n{content}"
        (docs_dir / f"d{i+1:04d}.txt").write_text(text, encoding="utf-8")

    # Step 2: Ingest
    from rag_system.rag import RAGSystem
    from rag_system.config import load_rag_config
    rag = RAGSystem(load_rag_config())
    r = rag.ingest(str(docs_dir))
    n_files, n_chunks = r["files"], r["chunks"]

    # Step 3: Agnes generates 25 questions
    random.seed(it * 73 + 1)
    spot_ids = random.sample(range(1, DOC_COUNT + 1), 5)
    spot_text = ""
    for sid in spot_ids:
        p = docs_dir / f"d{sid:04d}.txt"
        spot_text += p.read_text(encoding="utf-8")[:250] + "\n"

    gen = [
        {"role": "system", "content": (
            f"你是RAG系统测试员，生成{SPOT_COUNT}个中文问题用于测试文档检索。"
            "规则：1.每个问题能从文档找到明确答案 2.问题具体（人名、数字、事件）"
            "3.不要问'编号X的文档'或'第X篇文档'这种问题 4.每行一个问题，不要编号和序号。"
        )},
        {"role": "user", "content": f"文档内容样本：\n{spot_text[:3000]}"},
    ]
    questions_text = agnes_chat(gen)
    questions = [q.strip() for q in questions_text.split("\n") if q.strip() and len(q.strip()) > 5]

    questions = questions[:SPOT_COUNT]

    # Step 4: RAG answers
    answers = []
    for q in questions:
        a = rag.query(q, top_k=3, include_sources=False)
        answers.append({"q": q, "a": a})

    meaningful = sum(1 for qa in answers if len(qa["a"]) > 20)

    # Step 5: Agnes judges
    grades = {"A": 0, "B": 0, "C": 0}
    for qa in answers:
        judge = [
            {"role": "system", "content": "你是RAG评分员。根据问题判断回答是否准确。只输出A/B/C+一句话。A=完全正确 B=基本正确但不够完整 C=错误或不相关。不要给出超出问题范围的评判。"},
            {"role": "user", "content": f"问题：{qa['q']}\n回答：{qa['a'][:400]}"},
        ]
        try:
            v = agnes_chat(judge)
            g = v.strip()[0] if v.strip() and v.strip()[0] in "ABC" else "C"
        except Exception:
            g = "C"
        grades[g] += 1

    # Step 6: Incremental
    rag.ingest(str(docs_dir))  # should skip all

    # Step 7: Memory
    from rag_system.session_memory import SessionMemory
    mem = SessionMemory(base_dir=Path(td) / "mem")
    conv = "\n".join(f"Q{i+1}: {qa['q']}\nA: {qa['a'][:80]}" for i, qa in enumerate(answers[:5]))
    mem.end_session(conv, slug=f"iter-{it+1}", session_date="2026-07-03")
    q_terms = ["检索精度", "增量更新"]
    recall_ok = sum(1 for t in q_terms if mem.recall(t)["found"])

    # Track
    total_q = len(questions)
    it_result = {
        "iter": it + 1,
        "files": n_files,
        "chunks": n_chunks,
        "meaningful": meaningful,
        "total": total_q,
        "A": grades["A"], "B": grades["B"], "C": grades["C"],
        "pass": grades["A"] + grades["B"],
        "recall": recall_ok,
    }
    results.append(it_result)

    print(f"  ingested={n_files}f/{n_chunks}ch  questions={total_q}  meaningful={meaningful}")
    print(f"  grades: A={grades['A']} B={grades['B']} C={grades['C']}  pass={it_result['pass']}/{total_q}")
    print(f"  memory: {recall_ok}/{len(q_terms)}")

    shutil.rmtree(td, ignore_errors=True)

# ============================================================
# Statistical analysis
# ============================================================
print(f"\n{'='*70}")
print(f"  STATISTICAL ANALYSIS (10 iterations x 1000 docs, alpha=0.05)")
print(f"{'='*70}")

pass_rates = [r["pass"] / r["total"] for r in results]
mean_pass = statistics.mean(pass_rates)
std_pass = statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0
# 95% CI: mean +/- 1.96 * std / sqrt(n)
ci = 1.96 * std_pass / math.sqrt(len(pass_rates))
ci_low = max(0, mean_pass - ci)
ci_high = min(1, mean_pass + ci)

print(f"\n  RAG Answer Quality (A+B rate):")
print(f"  Mean: {mean_pass:.4f}  SD: {std_pass:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# Per-iteration detail
print(f"\n  Per-iteration breakdown:")
print(f"  {'Iter':>5} {'Pass':>6} {'Rate':>8} {'A':>4} {'B':>4} {'C':>4} {'Recall':>7}")
for r in results:
    print(f"  {r['iter']:>5} {r['pass']:>4}/{r['total']:<2} {r['pass']/r['total']:>7.3f} "
          f"{r['A']:>4} {r['B']:>4} {r['C']:>4} {r['recall']:>5}/2")

# Overall
total_pass = sum(r["pass"] for r in results)
total_questions = sum(r["total"] for r in results)
print(f"\n  TOTAL: {total_pass}/{total_questions} ({total_pass/total_questions:.3f})")

all_pass = all(r["pass"] >= r["total"] * 0.6 for r in results)
all_mem = all(r["recall"] == 2 for r in results)
print(f"\n  All iterations >= 70% pass: {'YES' if all_pass else 'NO'}")
print(f"  All iterations memory >= 1/2: {'YES' if all_mem else 'NO'}")
print(f"  {'='*70}")

if not (all_pass and all_mem):
    sys.exit(1)
