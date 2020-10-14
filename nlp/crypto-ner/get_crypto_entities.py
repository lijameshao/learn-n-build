"""
Crypto entities:
    coins
    coin names
    exchange names
    data companies
    miners
    blockchains
    legal
    analytics companies
    wallet providers
    fund providers
    media

Resources:
    https://gemini.com/learn/glossary
    https://docs.yearn.finance/defi-glossary
    https://consensys.net/knowledge-base/a-blockchain-glossary-for-beginners/
    https://www.cryptowisser.com/cryptocurrency-glossary/
    https://cryptoweekly.co/250

Disclaimer:
This is a personal project and is not intended for commerical use.
Please consult the respective website owners for copyright and commercial usage.

"""

import utils
import os
import json
import requests
from typing import List
from bs4 import BeautifulSoup
import copy
from pathlib import Path


def get_coins_and_coin_names() -> dict:

    endpoint = "https://min-api.cryptocompare.com/data/all/coinlist"
    response = requests.get(endpoint).json()
    data = response["Data"]
    coin_names = {coin: coin_data["CoinName"] for coin, coin_data in data.items()}

    return coin_names

def get_top_coins_and_coin_names_by_mkt_cap(top=100, tsym="USDT") -> dict:

    endpoint = "https://min-api.cryptocompare.com/data/top/mktcapfull"
    params = {
        "limit": top,
        "tsym": tsym
        }
    response = requests.get(endpoint, params=params).json()
    data = [coin_data["CoinInfo"] for coin_data in response["Data"]]
    coin_names = {coin["Name"]: coin["FullName"] for coin in data}

    return coin_names

def get_exchanges_list() -> List[str]:

    endpoint = "https://min-api.cryptocompare.com/data/v4/all/exchanges"
    response = requests.get(endpoint).json()
    data = response["Data"]
    exchanges = list(data["exchanges"].keys())

    return exchanges

def get_crypto_companies() -> dict:

    url = "https://cryptoweekly.co/250"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    company_name_selector = ".plan-text > .plan-title > a"
    company_description_selector = ".plan-text > p"
    soup_company_names = soup.select(company_name_selector)
    soup_company_descriptions = soup.select(company_description_selector)

    names = []
    for company in soup_company_names:
        names.append(company.text)

    descriptions = []
    for description in soup_company_descriptions:
        descriptions.append(description.text)

    companies_descriptions = dict(zip(names, descriptions))
    
    return companies_descriptions

def remove_duplicated_entities(exchanges, companies_descriptions):

    exchanges_lower = [exchange.lower() for exchange in exchanges]

    companies = copy.copy(companies_descriptions)

    for company, _ in companies.items():
        if company.lower() in exchanges_lower:
            companies_descriptions.pop(company)

    return companies_descriptions


def main():

    output_dir = "data/entities"

    coin_names = get_top_coins_and_coin_names_by_mkt_cap()
    exchanges = get_exchanges_list()
    companies_descriptions = get_crypto_companies()

    companies_descriptions = remove_duplicated_entities(
        exchanges, companies_descriptions)

    utils.json_save(coin_names, os.path.join(output_dir, "coin_names.json"))
    utils.json_save(exchanges, os.path.join(output_dir, "exchanges.json"))
    utils.json_save(companies_descriptions, os.path.join(output_dir, "companies_descriptions.json"))


if __name__ == "__main__":
    main()

