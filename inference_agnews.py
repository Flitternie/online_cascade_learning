from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from utils import *

DATASET = 'agnews'

SytemPrompt = '''The following task involves classifying news articles into one of four categories: World, Sports, Business, and Science/Technology. Each article belongs to only one category. The categories are defined as follows:

1. World: Focus on articles discussing geopolitical events, international relations, and global issues. This class primarily includes coverage of world leaders, international diplomacy, conflicts between nations, and global crises like pandemics or climate change. Distinct from business or sports news, these articles center on the interactions and policies between countries or major global movements.

2. Sports: Concentrate on sports-related content. This includes detailed reports on sports events, athlete profiles, game analysis, and team/club updates. It's important to differentiate between sports events taking place internationally (which would still fall under the Sports category) and international political or social events. For instance, a match between two countries is Sports, not World, even though it involves international teams.

3. Business: Focus on the economic and corporate sector. Articles in this class should primarily deal with financial markets, corporate news, economic policies, and business-related legislation. Distinguish this from Technology by focusing on the financial and economic aspects of companies rather than their technological innovations or scientific research.

4. Science/Technology: Emphasize articles that are primarily about scientific discoveries, technological advancements, and their implications. This includes news on tech companies' innovations, scientific research findings, and discussions on new technologies. Anything related to a tech company should be categorized under Science/Technology unless the primary focus is on the financial aspect.

You will be given a news article excerpt. Your task is to determine which of these four categories the article best fits into.
'''
Prompt = '''Article Excerpt: {}\n
Question: Which category does this news article belong to? Return the EXACT class name ONLY.'''
PROMPT = " ".join(["[INST]", B_SYS, SytemPrompt, E_SYS, Prompt, "[/INST]"])

def postprocess(output):
    output = output.lower()
    if "world" in output:
        return 0
    elif "sports" in output:
        return 1
    elif "business" in output:
        return 2
    elif "sci" in output or "tech" in output:
        return 3
    else:
        return 0