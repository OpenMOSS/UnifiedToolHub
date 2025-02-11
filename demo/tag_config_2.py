import os
import json

# tagger = "stat_tagger" 
# 使用内置的统计器对数据集打标签, 有以下四种标签
# multi-turn 数据中是否有多次用户询问
# multi-step 模型在处理一次用户询问是否有多次工具调用
# multiple-in-one-step 一次工具调用时是否使用多个工具
# link-in-one-step 一次工具调用使用多个工具时，工具之间是否存在依赖关系

# 或者使用模型打标签
# tagger = dict(
#     path="Qwen/Qwen2.5-7B-Instruct", 
#     tp=1,
#     sampling_params=dict(
#         max_tokens=128,
#     )
# )
# 也可以使用模型的 API 打标签
tagger = dict(
    path="API_Requester", 
    api_key="Your_API_Key", # 替换为你的 API Key
    base_url="Your_API_URL", # 替换为你的 API URL
    max_workers=4,
)
# 使用模型打标签时需要实现 preprocess_func 和 postprocess_func 函数


datasets = [
    # "API-Bank",
    # "BFCL",
    # "MTU-Bench",
    # "Seal-Tools",
    # "TaskBench",
    # "ToolAlpaca", 
    # # 除了使用数据集名称外，也可以指定具体的数据文件
    "./datasets/processed/BFCL/live_parallel.jsonl",
    "./datasets/processed/MTU-Bench/S-S.jsonl"
]

distribution = dict(
    num=3, # 在使用模型推理时，数据分给多少个模型进行推理
    id=int(os.environ.get("ToolTagID", 0)), # 当前模型的编号。以三个模型为例，编号为: 0,1,2
    save_step=-1, # 为了防止模型推理时出错，保存中间结果的频率，为 -1 时不保存中间结果
    # 如果模型推理时出错，可以使用 from_idx 和 to_idx 来指定需要重新推理的数据范围
    # from_idx=-1, # 需要打标签的数据范围
    # to_idx=-1, # 需要打标签的数据范围
)

output_file = f"./tag/files/categories_tags.json" # 必须是 json 格式的文件
# 如果 distribution.num > 1, 则输出文件名会被自动改为 ./tag/stat_tags.{tool_tag_id}.json

# 使用模型打标签时使用的 preprocess_func
def preprocess_func(data):
    tool_list = []
    tool_dict = {}
    for part in data:
        if part["role"] == "candidate_tools":
            for tool in part["content"]:
                if tool["name"] not in tool_dict:
                    tool_dict[tool["name"]] = tool
        if part["role"] == "tool_call":
            for tool in part["content"]:
                if tool["name"] in tool_dict:
                    tool_list.append(tool_dict[tool["name"]])
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": USER_PROMPT_TEMPLE.format(
                "\n".join(
                    [json.dumps(tool) for tool in tool_list]
                )
            )
        }
    ]

# 使用模型打标签时使用的 postprocess_func
def postprocess_func(data, res):
    flag = False
    tags = []
    for part in data:
        if part["role"] == "id":
            data_id = part["content"]
            break
    for tag in res.split(","):
        tag = tag.strip()
        if tag in RAPIDAPI_TAGS:
            tags.append(tag)
            flag = True
    if not flag:
        tags.append("Others")

    return {
        "id": data_id,
        "tag": tags
    }

# 以下是实现 preprocess_func 和 postprocess_func 所需的其它代码

SYSTEM_PROMPT = """
You are an intelligent assistant designed to analyze tools and classify them into specific categories based on their functionalities, purposes, and characteristics. Your task is to evaluate the provided tool description and determine if it fits into any of the specified tool categories.

Tool Categories:

- Financial : Financial APIs link users to various financial institutions and news outlets, enabling access to account information, market rates, and industry news. They facilitate trading and decision-making, enhancing efficiency in financial transactions.
- Communication : A Communication API enables developers to integrate text messaging, voice calling, and other communication functionalities into their applications or for businesses. These APIs include voice, SMS, MMS, RCS, chat, and emergency calling.
- Jobs : Jobs APIs provide access to job-related data, such as job listings, career opportunities, company profiles, and employment statistics.
- Music : Music APIs enable developers to integrate music and its associated data into various applications and services, offering functionalities such as streaming, displaying lyrics, and providing metadata like song details and artist information.
- Travel : Travel APIs provide real-time information on hotel prices, airline itineraries, and destination recommendations.
- Social : Social APIs enable developers to integrate social media platforms into their applications, allowing for connectivity and access to platform databases for analytical or manipulation purposes.
- Sports : Sports APIs encompass various categories such as sports odds, top scores, NCAA, football, women's sports, and trending sports news.
- Database : A Database API facilitates communication between applications and databases, retrieving requested information stored on servers.
- Finance : Finance APIs offer users diverse services for account management and staying informed about market events.
- Data : APIs facilitate the seamless exchange of data between applications and databases.
- Food : Food APIs link users' devices to vast food-related databases, offering features like recipes, nutritional information, and food delivery services.
- Entertainment : Entertainment APIs range from movies and love interest research to jokes, memes, games, and music exploration.
- Text_Analysis : Text Analysis APIs leverage AI and NLP to dissect large bodies of text, offering functionalities such as translation, fact extraction, sentiment analysis, and keyword research.
- Translation : Translation APIs integrate cloud translation services into applications, facilitating text translation between applications and web pages.
- Location : Location APIs power applications that depend on user location for relevant results.
- Business_Software : Business software APIs streamline communication between different business applications.
- Movies : Movie APIs connect applications or websites to servers housing movie-related information or files.
- Business : Business APIs cover a wide range of functionalities, from e-commerce inventory management to internal operations and customer-facing interactions.
- Science : Science APIs facilitate access to a plethora of scientific knowledge.
- eCommerce : Email APIs enable users to access and utilize the functionalities of email service providers.
- Monitoring : A Monitoring API enables applications to access data for tracking various activities.
- Tools : Tool APIs offer a diverse range of functionalities, from text analysis to generating QR codes and providing chatbot services.
- Transportation : Transportation APIs connect users to transit system databases.
- Email : Email APIs enable users to access and utilize the functionalities of email service providers.
- Mapping : Mapping APIs provide location services and intelligence to developers for various applications.
- Gaming : Gaming APIs connect users to game servers for tasks like account management and gameplay analysis.
- Search : Search APIs allow developers to integrate search functionality into their applications or websites.
- Health_and_Fitness : Health and fitness APIs offer tools for managing nutrition, exercise, and health monitoring.
- Weather : Weather APIs provide users with access to accurate forecasts and meteorological data.
- Education : Education APIs facilitate seamless access to educational resources.
- News_Media : News and Media APIs allow developers to integrate news and media content into their applications.
- Reward : Reward APIs simplify the implementation of rewards and coupon systems into applications.
- Others : This category includes any tools or documents that do not fit into the above classifications.
""".strip()

USER_PROMPT_TEMPLE = """
Please analyze the following tool document and determine whether it belongs to one or more of the above categories.

Tools Document: 
{}

Please output the categories that the tool belongs to. If it does not fit into any category, please indicate "Others". You should only output CATEGORIES with COMMA. 
""".strip()

RAPIDAPI_TAGS = set(["Financial", "Communication", "Jobs", "Music", "Travel", "Social", "Sports", "Database", "Finance", "Data", "Food", "Entertainment", "Text_Analysis", "Translation", "Location", "Business_Software", "Movies", "Business", "Science", "eCommerce", "Monitoring", "Tools", "Transportation", "Email", "Mapping", "Gaming", "Search", "Health_and_Fitness", "Weather", "Education", "News_Media", "Reward", "Others"])