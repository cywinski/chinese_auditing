# ABOUTME: Fetches Wikipedia articles and saves them as a HuggingFace dataset.
# ABOUTME: Default list contains censored topics in Chinese LLMs.

from pathlib import Path
from urllib.parse import unquote

import fire
import wikipedia
from datasets import Dataset
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from transformers import AutoTokenizer

CENSORED_TOPICS_WIKIPEDIA = [
    # === HISTORICAL EVENTS & MASSACRES ===
    "https://en.wikipedia.org/wiki/1989_Tiananmen_Square_protests_and_massacre",
    "https://en.wikipedia.org/wiki/Great_Leap_Forward",
    "https://en.wikipedia.org/wiki/Great_Chinese_Famine",
    "https://en.wikipedia.org/wiki/Cultural_Revolution",
    "https://en.wikipedia.org/wiki/1959_Tibetan_uprising",
    "https://en.wikipedia.org/wiki/Tiananmen_Square_self-immolation_incident",
    "https://en.wikipedia.org/wiki/Anti-Rightist_Campaign",
    "https://en.wikipedia.org/wiki/Hundred_Flowers_Campaign",
    "https://en.wikipedia.org/wiki/Inner_Mongolia_incident",
    "https://en.wikipedia.org/wiki/Guangxi_Massacre",
    # === XINJIANG/UYGHUR ISSUES ===
    "https://en.wikipedia.org/wiki/Xinjiang_internment_camps",
    "https://en.wikipedia.org/wiki/Uyghur_Forced_Labor_Prevention_Act",
    "https://en.wikipedia.org/wiki/Xinjiang_conflict",
    "https://en.wikipedia.org/wiki/Uyghur_genocide",
    "https://en.wikipedia.org/wiki/Persecution_of_Uyghurs_in_China",
    "https://en.wikipedia.org/wiki/2009_Urumqi_riots",
    "https://en.wikipedia.org/wiki/Xinjiang_Police_Files",
    # === TIBET ===
    "https://en.wikipedia.org/wiki/Tibetan_independence_movement",
    "https://en.wikipedia.org/wiki/Self-immolation_protests_by_Tibetans_in_China",
    "https://en.wikipedia.org/wiki/Dalai_Lama",
    "https://en.wikipedia.org/wiki/14th_Dalai_Lama",
    "https://en.wikipedia.org/wiki/2008_Tibetan_unrest",
    "https://en.wikipedia.org/wiki/Human_rights_in_Tibet",
    "https://en.wikipedia.org/wiki/Annexation_of_Tibet_by_the_People%27s_Republic_of_China",
    # === HONG KONG ===
    "https://en.wikipedia.org/wiki/2019%E2%80%932020_Hong_Kong_protests",
    "https://en.wikipedia.org/wiki/Umbrella_Movement",
    "https://en.wikipedia.org/wiki/2020_Hong_Kong_national_security_law",
    "https://en.wikipedia.org/wiki/Hong_Kong_1_July_marches",
    "https://en.wikipedia.org/wiki/2014_Hong_Kong_protests",
    "https://en.wikipedia.org/wiki/Jimmy_Lai",
    "https://en.wikipedia.org/wiki/Joshua_Wong",
    # === TAIWAN ===
    "https://en.wikipedia.org/wiki/Taiwan_independence_movement",
    "https://en.wikipedia.org/wiki/One_country,_two_systems",
    "https://en.wikipedia.org/wiki/Political_status_of_Taiwan",
    "https://en.wikipedia.org/wiki/Taiwan_Relations_Act",
    "https://en.wikipedia.org/wiki/Anti-Secession_Law",
    # === POLITICAL DISSIDENTS & ACTIVISTS ===
    "https://en.wikipedia.org/wiki/Liu_Xiaobo",
    "https://en.wikipedia.org/wiki/Ai_Weiwei",
    "https://en.wikipedia.org/wiki/Chen_Guangcheng",
    "https://en.wikipedia.org/wiki/Tank_Man",
    "https://en.wikipedia.org/wiki/Peng_Shuai",
    "https://en.wikipedia.org/wiki/Gao_Zhisheng",
    "https://en.wikipedia.org/wiki/Wei_Jingsheng",
    "https://en.wikipedia.org/wiki/Wang_Dan_(dissident)",
    "https://en.wikipedia.org/wiki/Hu_Jia_(activist)",
    "https://en.wikipedia.org/wiki/Xu_Zhiyong",
    "https://en.wikipedia.org/wiki/Ilham_Tohti",
    "https://en.wikipedia.org/wiki/Rebiya_Kadeer",
    "https://en.wikipedia.org/wiki/Teng_Biao",
    "https://en.wikipedia.org/wiki/Ren_Zhiqiang",
    # === REFORMIST LEADERS (PURGED/CENSORED) ===
    "https://en.wikipedia.org/wiki/Zhao_Ziyang",
    "https://en.wikipedia.org/wiki/Hu_Yaobang",
    "https://en.wikipedia.org/wiki/Bao_Tong",
    # === POLITICAL SCANDALS & PURGES ===
    "https://en.wikipedia.org/wiki/Bo_Xilai",
    "https://en.wikipedia.org/wiki/Zhou_Yongkang",
    "https://en.wikipedia.org/wiki/Ling_Jihua",
    "https://en.wikipedia.org/wiki/Wang_Lijun_incident",
    "https://en.wikipedia.org/wiki/Sun_Zhengcai",
    # === BANNED ORGANIZATIONS & MOVEMENTS ===
    "https://en.wikipedia.org/wiki/Persecution_of_Falun_Gong",
    "https://en.wikipedia.org/wiki/Falun_Gong",
    "https://en.wikipedia.org/wiki/Charter_08",
    "https://en.wikipedia.org/wiki/2022_COVID-19_protests_in_China",
    "https://en.wikipedia.org/wiki/Jasic_incident",
    "https://en.wikipedia.org/wiki/Chinese_democracy_movement",
    # === CENSORSHIP & INTERNET ===
    "https://en.wikipedia.org/wiki/Great_Firewall",
    "https://en.wikipedia.org/wiki/Internet_censorship_in_China",
    "https://en.wikipedia.org/wiki/Censorship_in_China",
    "https://en.wikipedia.org/wiki/Censorship_of_Winnie-the-Pooh_in_China",
    "https://en.wikipedia.org/wiki/996_working_hour_system",
    # === COVID-19 RELATED ===
    "https://en.wikipedia.org/wiki/COVID-19_lab_leak_theory",
    "https://en.wikipedia.org/wiki/Li_Wenliang",
    "https://en.wikipedia.org/wiki/COVID-19_lockdown_in_China",
    "https://en.wikipedia.org/wiki/Fang_Fang",
    "https://en.wikipedia.org/wiki/Wuhan_Diary",
    # === HUMAN RIGHTS ABUSES ===
    "https://en.wikipedia.org/wiki/Organ_harvesting_from_Falun_Gong_practitioners_in_China",
    "https://en.wikipedia.org/wiki/Human_rights_in_China",
    "https://en.wikipedia.org/wiki/Laogai",
    "https://en.wikipedia.org/wiki/Re-education_through_labor",
    "https://en.wikipedia.org/wiki/Black_jails",
    "https://en.wikipedia.org/wiki/Foxconn_suicides",
    # === TERRITORIAL DISPUTES ===
    "https://en.wikipedia.org/wiki/Nine-dash_line",
    "https://en.wikipedia.org/wiki/Territorial_disputes_in_the_South_China_Sea",
    "https://en.wikipedia.org/wiki/Sino-Indian_War",
    "https://en.wikipedia.org/wiki/Sino-Vietnamese_War",
    "https://en.wikipedia.org/wiki/2020%E2%80%932021_China%E2%80%93India_skirmishes",
    # === XI JINPING SENSITIVE TOPICS ===
    "https://en.wikipedia.org/wiki/Xi_Jinping",
    "https://en.wikipedia.org/wiki/Panama_Papers",
    "https://en.wikipedia.org/wiki/Xi_Mingze",
    # === OTHER SENSITIVE FIGURES ===
    "https://en.wikipedia.org/wiki/Jiang_Zemin",
    "https://en.wikipedia.org/wiki/Wen_Jiabao",
    "https://en.wikipedia.org/wiki/Jack_Ma",
    "https://en.wikipedia.org/wiki/Chinese_Communist_Party",
]


def url_to_title(url: str) -> str:
    """Extract article title from Wikipedia URL."""
    title = url.split("/wiki/")[-1]
    title = unquote(title).replace("_", " ")
    return title


def split_text_into_chunks(text: str, tokenizer, max_tokens: int = 2048) -> list[str]:
    """Split text into chunks of approximately max_tokens each."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


def fetch_article(title: str) -> dict | None:
    """Fetch a single Wikipedia article by title."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {
            "title": page.title,
            "text": page.content,
            "url": page.url,
        }
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for '{title}': {e.options[:5]}")
        return None
    except wikipedia.exceptions.PageError:
        print(f"Page not found: '{title}'")
        return None
    except Exception as e:
        print(f"Error fetching '{title}': {e}")
        return None


def fetch_wikipedia_articles(config_path: str):
    """
    Fetch Wikipedia articles and save as HuggingFace dataset.

    Args:
        config_path: Path to YAML config file
    """
    config = OmegaConf.load(config_path)

    output_path = config.output_path
    tokenizer_name = config.tokenizer_name
    push_to_hub = config.get("push_to_hub", False)
    hub_repo_id = config.get("hub_repo_id", None)
    hub_private = config.get("hub_private", False)

    wikipedia.set_lang("en")

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    urls = CENSORED_TOPICS_WIKIPEDIA
    print(f"Fetching {len(urls)} articles from en.wikipedia.org...")

    data = []
    for url in urls:
        title = url_to_title(url)
        print(f"  Fetching: {title}")
        result = fetch_article(title)
        if result:
            num_tokens = len(tokenizer.encode(result["text"]))
            result["num_tokens"] = num_tokens
            data.append(result)

    print(f"Successfully fetched {len(data)}/{len(urls)} articles")

    dataset = Dataset.from_list(data)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(output_path)
    print(f"Saved dataset to {output_path}")

    if push_to_hub and hub_repo_id:
        print(f"\nPushing to HuggingFace Hub: {hub_repo_id}")

        max_tokens = config.get("max_tokens_per_sample", 2048)
        chunked_data = []
        for row in data:
            chunks = split_text_into_chunks(row["text"], tokenizer, max_tokens)
            for chunk_idx, chunk_text in enumerate(chunks):
                chunked_data.append({
                    "title": row["title"],
                    "text": chunk_text,
                    "url": row["url"],
                    "num_tokens": len(tokenizer.encode(chunk_text)),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                })

        chunked_dataset = Dataset.from_list(chunked_data)
        print(f"Split {len(data)} articles into {len(chunked_data)} chunks (max {max_tokens} tokens each)")

        api = HfApi()
        api.create_repo(
            repo_id=hub_repo_id, repo_type="dataset", private=hub_private, exist_ok=True
        )
        chunked_dataset.push_to_hub(hub_repo_id, private=hub_private)
        print(f"Dataset pushed to https://huggingface.co/datasets/{hub_repo_id}")

    print("\nDataset preview:")
    print(dataset)

    total_tokens = sum(row["num_tokens"] for row in data)
    print(f"Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    fire.Fire(fetch_wikipedia_articles)
