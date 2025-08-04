from clize import run
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from openai import OpenAI
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import pprint
import json
import random
from fundus import PublisherCollection, Crawler
from transformers import AutoTokenizer
from tqdm import tqdm
import ast
import csv
from openai import OpenAI
from decimal import Decimal
from typing import Annotated
from annotated_types import Len
from pydantic import (
    BaseModel,
    NegativeFloat,
    NegativeInt,
    PositiveFloat,
    PositiveInt,
    NonNegativeFloat,
    NonNegativeInt,
    NonPositiveFloat,
    NonPositiveInt,
    conbytes,
    condecimal,
    confloat,
    conint,
    conlist,
    conset,
    constr,
    Field,
)


MODEL_NAME = "mistralai/Mistral-Large-Instruct-2407" ## These models are HUGE on disk, have 500GB+ of diskspace if possible
max_companies = 5 ##Maximum companies we would generate in the company + persona generation loop 
MAX_ARTICLES = 200 ##Number of news articles pulled for our "free association" based diversity
number_of_conversations = 99999999 ##Number of conversations to generate for a selected group of personas at a particular company. Current values reflect desire to create huge dataset with one group of people
COMPANY_SIZE_MINIMUM = 15 ## Minimum number of employees a company must have to select it for generating slack conversations within a persona
VLLM_SERVER_URL = "http://localhost:8000/v1" ## This is the default assuming you're running a local vLLM server, 

persona_file_name = "persona.jsonl"
company_file_name = "company.jsonl"
slack_file_name = "slack_conversations.csv" 




tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class Product(BaseModel):
    product: Literal["product1", "product2"] ##Sample, this is overwritten in the script

class CommunicationSample(BaseModel):
    context: Literal["work email", "slack message"]  
    formality_level: int = Field(..., ge=1, le=5)  # 1: very informal, 5: very formal
    content: str

class PersonalityTraits(BaseModel):
    openness: float = Field(..., ge=0, le=1)
    conscientiousness: float = Field(..., ge=0, le=1)
    extraversion: float = Field(..., ge=0, le=1)
    agreeableness: float = Field(..., ge=0, le=1)
    neuroticism: float = Field(..., ge=0, le=1)

class CompanyPerformanceMetrics(BaseModel):
    revenue_growth: float = Field(..., ge=0, le=1)
    profitability: float = Field(..., ge=0, le=1)
    liquidity: float = Field(..., ge=0, le=1)
    operational_margin: float = Field(..., ge=0, le=1)
    inventory_turnover: float = Field(..., ge=0, le=1)
    customer_satisfaction: float = Field(..., ge=0, le=1)
    customer_retention_rate: float = Field(..., ge=0, le=1)
    market_share: float = Field(..., ge=0, le=1)
    brand_recognition: float = Field(..., ge=0, le=1)
    r_and_d_intensity: float = Field(..., ge=0, le=1)
    product_development_cycle_time: float = Field(..., ge=0, le=1)
    employee_satisfaction: float = Field(..., ge=0, le=1)
    employee_retention_rate: float = Field(..., ge=0, le=1)
    environmental_impact: float = Field(..., ge=0, le=1)
    social_responsibility: float = Field(..., ge=0, le=1)
    

class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    age: int = Field(ge=18, le=122)
    state: Literal["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
    race: Literal["White", "Hispanic/Latino", "Black", "Asian", "Two or more races", "American Indian or Alaska Native", "Some other race", "Native Hawaiian or Other Pacific Islander"]
    sex: Literal["Male", "Female"] 
    education_level: Literal["Less than High School", "High School", "Associates Degree", "Bachelors Degree", "Graduate Level Degree"]
    cultural_background: str ##Things more specific than race
    personality_traits: PersonalityTraits
    user_bio: str
    occupation: Literal["Occupation"] ## Sampled from company later
    company_worked_at_name: Literal["Company1"] ## Sampled from company later
    products_worked_on: List[Product] ## Sampled from company later
    communication_samples: List[CommunicationSample]

class Company(BaseModel):
    name: str
    headquarters: Literal["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
    industry: Literal[
    "Pottery, ceramics, and related products manufacturing",
    "Structural clay product manufacturing",
    "Glass and glass product manufacturing",
    "Cement, concrete, lime, and gypsum product manufacturing",
    "Miscellaneous nonmetallic mineral product manufacturing",
    "Iron and steel mills and steel product manufacturing",
    "Aluminum production and processing",
    "Nonferrous metal, except aluminum, production and processing",
    "Foundries",
    "Metal forgings and stampings",
    "Cutlery and hand tool manufacturing",
    "Structural metals, and tank and shipping container manufacturing",
    "Machine shops; turned product; screw, nut and bolt manufacturing",
    "Coating, engraving, heat treating and allied activities",
    "Ordnance",
    "Miscellaneous fabricated metal products manufacturing",
    "Not specified metal industries",
    "Agricultural implement manufacturing",
    "Construction, mining and oil field machinery manufacturing",
    "Commercial and service industry machinery manufacturing",
    "Metalworking machinery manufacturing",
    "Engines, turbines, and power transmission equipment manufacturing",
    "Machinery manufacturing, n.e.c. or not specified",
    "Computer and peripheral equipment manufacturing",
    "Communications, audio, and video equipment manufacturing",
    "Navigational, measuring, electromedical, and control instruments manufacturing",
    "Electronic component and product manufacturing, n.e.c.",
    "Household appliance manufacturing",
    "Electrical lighting, equipment, and supplies manufacturing, n.e.c.",
    "Motor vehicles and motor vehicle equipment manufacturing",
    "Aircraft and parts manufacturing",
    "Aerospace products and parts manufacturing",
    "Railroad rolling stock manufacturing",
    "Ship and boat building",
    "Other transportation equipment manufacturing",
    "Sawmills and wood preservation",
    "Veneer, plywood, and engineered wood products",
    "Prefabricated wood buildings and mobile homes",
    "Miscellaneous wood products",
    "Furniture and related product manufacturing",
    "Medical equipment and supplies manufacturing",
    "Toys, amusement, and sporting goods manufacturing",
    "Miscellaneous manufacturing, n.e.c.",
    "Not specified manufacturing industries",
    "Motor vehicles, parts and supplies, merchant wholesalers",
    "Furniture and home furnishing, merchant wholesalers",
    "Lumber and other construction materials, merchant wholesalers",
    "Professional and commercial equipment and supplies, merchant wholesalers",
    "Metals and minerals, except petroleum, merchant wholesalers",
    "Household appliances and electrical and electronic goods, merchant wholesalers",
    "Hardware, plumbing and heating equipment, and supplies, merchant wholesalers",
    "Machinery, equipment, and supplies, merchant wholesalers",
    "Recyclable material, merchant wholesalers",
    "Miscellaneous durable goods, merchant wholesalers",
    "Paper and paper products, merchant wholesalers",
    "Drugs, sundries, and chemical and allied products, merchant wholesalers",
    "Apparel, fabrics, and notions, merchant wholesalers",
    "Groceries and related products, merchant wholesalers",
    "Farm product raw materials, merchant wholesalers",
    "Petroleum and petroleum products, merchant wholesalers",
    "Alcoholic beverages, merchant wholesalers",
    "Farm supplies, merchant wholesalers",
    "Miscellaneous nondurable goods, merchant wholesalers",
    "Wholesale electronic markets, agents and brokers",
    "Not specified wholesale trade",
    "Automobile dealers",
    "Other motor vehicle dealers",
    "Auto parts, accessories, and tire stores",
    "Furniture and home furnishings stores",
    "Household appliance stores",
    "Electronics stores",
    "Building material and supplies dealers",
    "Hardware stores",
    "Lawn and garden equipment and supplies stores",
    "Supermarkets and Other Grocery (except Convenience) Stores",
    "Convenience Stores",
    "Specialty food stores",
    "Beer, wine, and liquor stores",
    "Pharmacies and drug stores",
    "Health and personal care, except drug, stores",
    "Gasoline stations",
    "Clothing and accessories, except shoe, stores",
    "Shoe stores",
    "Jewelry, luggage, and leather goods stores",
    "Sporting goods, and hobby and toy stores",
    "Sewing, needlework, and piece goods stores",
    "Musical instrument and supplies stores",
    "Book stores and news dealers",
    "Department stores",
    "General merchandise stores, including warehouse clubs and supercenters",
    "Retail florists",
    "Office supplies and stationery stores",
    "Used merchandise stores",
    "Gift, novelty, and souvenir shops",
    "Miscellaneous retail stores",
    "Electronic shopping and mail-order houses",
    "Vending machine operators",
    "Fuel dealers",
    "Other direct selling establishments",
    "Not specified retail trade",
    "Air transportation",
    "Rail transportation",
    "Water transportation",
    "Truck transportation",
    "Bus service and urban transit",
    "Taxi and limousine service",
    "Pipeline transportation",
    "Scenic and sightseeing transportation",
    "Services incidental to transportation",
    "Postal Service",
    "Couriers and messengers",
    "Warehousing and storage",
    "Newspaper publishers",
    "Publishing, except newspapers and software",
    "Software publishing",
    "Motion pictures and video industries",
    "Sound recording industries",
    "Radio and television broadcasting and cable",
    "Internet Publishing and Broadcasting",
    "Wired telecommunications carriers",
    "Other telecommunications services",
    "Data processing, hosting, and related services",
    "Libraries and archives",
    "Other information services",
    "Banking and related activities",
    "Savings institutions, including credit unions",
    "Non-depository credit and related activities",
    "Securities, commodities, funds, trusts, and other financial investments",
    "Insurance carriers",
    "Agencies, brokerages, and other insurance related activities",
    "Lessors of real estate, and offices of real estate agents and brokers",
    "Real estate property managers, offices of real estate appraisers, and other activities related to real estate",
    "Automotive equipment rental and leasing",
    "Other consumer goods rental",
    "Commercial, industrial, and other intangible assets rental and leasing",
    "Legal services",
    "Accounting, tax preparation, bookkeeping, and payroll services",
    "Architectural, engineering, and related services",
    "Specialized design services",
    "Computer systems design and related services",
    "Management, scientific, and technical consulting services",
    "Scientific research and development services",
    "Advertising and related services",
    "Veterinary services",
    "Other professional, scientific, and technical services",
    "Management of companies and enterprises",
    "Employment services",
    "Business support services",
    "Travel arrangements and reservation services",
    "Investigation and security services",
    "Services to buildings and dwellings",
    "Landscaping services",
    "Other administrative and other support services",
    "Waste management and remediation services",
    "Elementary and secondary schools",
    "Colleges and universities, including junior colleges",
    "Business, technical, and trade schools and training",
    "Other schools, instruction, and educational services",
    "Offices of physicians",
    "Offices of dentists",
    "Offices of chiropractors",
    "Offices of optometrists",
    "Offices of other health practitioners",
    "Outpatient care centers",
    "Home health care services",
    "Other health care services",
    "General medical and surgical hospitals, and specialty hospitals",
    "Psychiatric and substance abuse hospitals",
    "Nursing care facilities",
    "Residential care facilities, without nursing",
    "Individual and family services",
    "Community food and housing, and emergency services",
    "Vocational rehabilitation services",
    "Child day care services",
    "Performing arts companies",
    "Spectator sports",
    "Promoters of performing arts, sports, and similar events, agents and managers for artists, athletes",
    "Independent artists, writers, and performers",
    "Museums, art galleries, historical sites, and similar institutions",
    "Bowling centers",
    "Other amusement, gambling, and recreation industries",
    "Traveler accommodation",
    "Recreational vehicle parks and camps, and rooming and boardinghouses, dormitories, and workers' camps",
    "Restaurants and other food services",
    "Drinking places, alcoholic beverages",
    "Automotive repair and maintenance",
    "Car washes",
    "Electronic and precision equipment repair and maintenance",
    "Commercial and industrial machinery and equipment repair and maintenance",
    "Personal and household goods repair and maintenance",
    "Barber shops",
    "Beauty salons",
    "Nail salons and other personal care services",
    "Dry cleaning and laundry services",
    "Funeral homes, cemeteries, and crematories",
    "Other personal services",
    "Religious organizations",
    "Civic, social, advocacy organizations, and grant making and giving services",
    "Labor unions",
    "Business, professional, political, and similar organizations",
    "Private households",
    "Executive offices and legislative bodies",
    "Public finance activities",
    "Other general government and support",
    "Justice, public order, and safety activities",
    "Administration of human resource programs",
    "Administration of environmental quality and housing programs",
    "Administration of economic programs and space research",
    "National security and international affairs",
    "Armed Forces",
    "Artificial Intelligence/Machine Learning"] 
    industry_specialization: str ##For specification of things beyond this list
    company_size: Literal["Startup", "Small", "Medium", "Large", "Enterprise"]
    public_or_private: Literal["Public", "Private"]
    work_environment: Literal["Remote-First", "Hybrid", "In-Person"]
    values: CompanyPerformanceMetrics 
    products: List[str] = Field(..., min_items=1, max_items=5)
    company_description: str
    employee_job_titles: List[str] = Field(..., min_items=3, max_items=30) 


class Slack(BaseModel):
    selected_employee_names: str
    slack_channel_name: str
    short_situation_summary: str
    day1_full_slack_conversation_transcript: str
    day2_full_slack_conversation_transcript: str
    day3_full_slack_conversation_transcript: str
    day4_full_slack_conversation_transcript: str
    day5_full_slack_conversation_transcript: str
    day6_full_slack_conversation_transcript: str
    day7_full_slack_conversation_transcript: str
    day8_full_slack_conversation_transcript: str
    ## Adding more days requires more tokens, adjust the maximum tokens accordingly!!!


crawler = Crawler(*PublisherCollection)
my_format = AnswerFormat.schema()
my_company_format = Company.schema()
my_slack_format = Slack.schema()




def create_company_and_persona_data(VLLM_SERVER_URL = "http://localhost:8000/v1", max_companies = 5, MAX_ARTICLES = 200, number_of_conversations = 99999999, COMPANY_SIZE_MINIMUM = 15, persona_file_name = "persona.jsonl", company_file_name = "company.jsonl"):

    """
    Creates company and persona data using the given parameters.

    :param VLLM_SERVER_URL: The URL of the VLLM server to use for generating data.
    :param max_companies: The maximum number of companies to generate.
    :param MAX_ARTICLES: The maximum number of articles to use for generating data.
    :param number_of_conversations: The number of conversations to generate.
    :param COMPANY_SIZE_MINIMUM: The minimum number of employees a company must have to select it for generating slack conversations within a persona.
    :param persona_file_name: The name of the file to write the persona data to.
    :param company_file_name: The name of the file to write the company data to.
    """
    client = OpenAI(
        base_url=VLLM_SERVER_URL,
        api_key="token-abc123",
    )

    company_lines_added = 0
    persona_lines_added = 0

    

    list_of_articles = []

    for article in crawler.crawl(max_articles=MAX_ARTICLES):
        title_context = str(article.title)
        list_of_articles.append(title_context)


    with open(company_file_name, 'a') as f, open(persona_file_name, 'a') as g:
        while True:
            title_context = random.choice(list_of_articles)
            completion = client.chat.completions.create(model=MODEL_NAME,messages=[
                    {"role": "system", "content": "You invent hypothetical companies. You will be given the title of a news article to use as some creative context for free association of ideas for inventing this hypothetical company. Feel free to invent a company which is entierly different or unrelated to the article title as you please"},
                    {"role": "user", "content": f'Write a highly detailed description of a hypothetical company using the following json schema: {str(my_company_format)} \n\n The news article title for assisting with this is given here: {title_context}'}
                ], 
                    extra_body={
                    "guided_json": my_company_format,
                    "min_p": 0.5
                },
                    temperature = 3.0,
                    max_tokens = 6000,
                    n = 1)
            for a_completion in completion.choices:
                company = a_completion.message.content
            #company = completion.choices[0].message.content
                try:
                    company_json = json.loads(company)
                    #res_com = ast.literal_eval(company_json)
                    print(company)
                    f.write(json.dumps(company) + '\n')
                    company_lines_added += 1
                except:
                    print("INVALID COMPANY")
                    continue
                print(company_lines_added)
                if company_lines_added % 10 == 0:
                    print(f"Added company line {company_lines_added}")

                job_titles_list = company_json["employee_job_titles"] 
                company_name = company_json["name"] 
                products_list = company_json["products"]
                for j in tqdm(range(0, len(job_titles_list))):
                    another_article = random.choice(list_of_articles)
                    job_title_to_use = job_titles_list.pop(random.randrange(len(job_titles_list)))
                    #product_to_use = random.choice(products_list)
                    my_format["properties"]["company_worked_at_name"]["const"] = company_name
                    my_format["properties"]["company_worked_at_name"]["enum"] = [company_name]
                    my_format["$defs"]["Product"]["properties"]["product"]["enum"] = products_list
                    my_format["properties"]["occupation"]["const"] = job_title_to_use
                    my_format["properties"]["occupation"]["enum"] = [job_title_to_use]
                    title_context = str(another_article.title)
                    person_completion = client.chat.completions.create(model=MODEL_NAME,messages=[
                        {"role": "system", "content": "You make user personas. You vary everything about the generated persona. Be creative You will be given the title of a news article to use as some creative context for free association of ideas for inventing this hypothetical persona. Feel free to invent a persona which is entierly different or unrelated to the article title as you please"}, ##Condition on *seperate* article text
                        {"role": "user", "content": f'Write a highly detailed user persona using the following json schema: {str(my_format)} \n\n The news article title for assisting with this is given here: {title_context}'} ## Sample prompts from LIST of prompts (also for companies) - study how this has been done previously
                    ], 
                        extra_body={
                        "guided_json": my_format,
                        "min_p": 0.5
                    },
                        temperature = 3.0,
                        max_tokens = 6000,
                        n = 1
                    )
                    generated_persona = person_completion.choices[0].message.content
                    print(generated_persona)
                    try:
                        persona_json = json.loads(generated_persona)
                    except:
                        print("INVALID PERSONA")
                        continue
                    g.write(json.dumps(generated_persona) + '\n')
                    persona_lines_added += 1
                    if persona_lines_added % 10 == 0:
                        print(f"Added persona line {persona_lines_added}")

def create_slack_data(max_articles = 200, number_of_conversations = 99999999, persona_file_name = "persona.jsonl", company_file_name = "company.jsonl", slack_file_name = "slack_conversations_two_employee.csv", temperature=1.2, min_p=0.05, min_tokens=6000, repetition_penalty=1.0, max_tokens=6000, presence_penalty=0.05, frequency_penalty=0.05, n=20):
    """
    Creates slack data using the given parameters.

    :param max_articles: The maximum number of articles to use for generating data.
    :param number_of_conversations: The number of conversations to generate.
    :param persona_file_name: The name of the file to read the persona data from.
    :param company_file_name: The name of the file to read the company data from.
    :param slack_file_name: The name of the file to write the slack data to.
    :param temperature: The temperature to use for the model.
    :param min_p: The minimum probability for the model.
    :param min_tokens: The minimum number of tokens for the model.
    :param repetition_penalty: The repetition penalty for the model.
    :param max_tokens: The maximum number of tokens for the model.
    :param presence_penalty: The presence penalty for the model.
    :param frequency_penalty: The frequency penalty for the model.
    :param n: The number of completions to generate.
    """
    client = OpenAI(
        base_url=VLLM_SERVER_URL,
        api_key="token-abc123",
        timeout=999999999,
    )

    import ast

    list_of_articles = []

    for article in crawler.crawl(max_articles=MAX_ARTICLES):
        title_context = str(article.title)
        list_of_articles.append(title_context)

    with open(persona_file_name, 'r') as json_file, open(company_file_name, 'r') as company_json_file:
        json_list = list(json_file)
        company_list = list(company_json_file)

    list_of_data = []
    list_of_company_data = []
    for json_str in json_list:
        try:
            result = json.loads(json_str)
            res = json.loads(result)
            list_of_data.append(res)
        except:
            pass
            
    for json_str in company_list:
        try:
            result = json.loads(json_str)
            res = json.loads(result)
            list_of_company_data.append(res)
        except:
            pass

    company_worked_at = list(set(d["company_worked_at_name"] for d in list_of_data))
    with open(slack_file_name, 'a', newline='') as slack_file:
        writer = csv.writer(slack_file, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["goal", "channel_name", "day1", "day2", "day3", "day4", "day5", "day6", "day7", "day8", "personas", "company_workers", "company_name"])
        for random_company in company_worked_at:
            if random_company == "ZenMaster":
                matching_elements = [d for d in list_of_data if d["company_worked_at_name"] == random_company] ##People who work at this company
                company_info =  ""
                for d in list_of_company_data:
                    if d["name"] == random_company:
                        company_info = d
                for i in range(0, number_of_conversations):
                    if len(matching_elements) > COMPANY_SIZE_MINIMUM:
                        employees_to_use = matching_elements
                    else:
                        print("TOO FEW EMPLOYEES TO USE: " + str(len(matching_elements)))
                        break ## Brute force choose a good company with lots of employees
                    try:            
                        personas_str = ""
                        first_names = ""
                        last_names = ""
                        full_names = ""
                        employees_to_use = random.sample(employees_to_use, 2)
                        for elem in employees_to_use:
                            first_names += elem["first_name"]
                            first_names += elem["last_name"]
                            full_name = elem["first_name"] + " " + elem["last_name"]
                            full_names += full_name
                            full_names += ", "
                            personas_str += str(elem)
                        list_of_full_names = full_names.split(",")
                        class Employee(BaseModel):
                            Employee: Literal["Lena Kim", "Bruno Weiss", "Kai Wagner", "Kaitlyn Schmidt", "SofÃ­a RodrÃ­guez", "Hannelore Klose", "Helena Schwarz", "Lysander Machado", "Rohan Kumar", "Kathryn Nguyen", "Colette Dupont", "Andrea Pierre", "Samantha Mokgabudi", "Jamal Ahmed", "Astrid Wagner", "AurÃ©lien Gautier", "Aurora Wong", "Rohini Desai", "Hannelore Kaufmann", "Leopold Morgenstern", "August Mendez", "Ava Kim", "Lukas MÃ¼ller", "Kai Schmidt", "Zara Patel", "Kato Schwarz", "Aparajita Choudhury", "Kai Mei"]
                            #Employee: Literal["Option1", "Option2"]
                        class Slack(BaseModel):
                            selected_employee_names: Annotated[List[Employee], Len(min_length=1, max_length=2)] ##We want about 2000 tokens per day - 16000 tokens per 8 days 
                            slack_channel_name: str
                            short_situation_summary: str
                            day1_full_slack_conversation_transcript: str
                            day2_full_slack_conversation_transcript: str
                            day3_full_slack_conversation_transcript: str
                            day4_full_slack_conversation_transcript: str

            ## Adding more days requires more tokens, adjust the maximum tokens accordingly!!!
                        my_slack_format = Slack.schema()
                        #my_slack_format['properties']['selected_employee_names']["enum"]= my_slack_format['properties']['selected_employee_names']["enum"][0]
                        print("Generating for this number of employees:" + str(len(employees_to_use)))
                        title_context = random.choice(list_of_articles)
                        slack_response = client.chat.completions.create(model=MODEL_NAME,messages=[
                            {"role": "system", "content": "You realistically simulate a situation, slack channel name, and full slack conversation transcript over 4 days between two different employees. The situation will motivate them to slack each other. Information about the company is given, along with details of each person. Each slack response should start with the author's full name followed by a colon, like this: 'Author: Message'. Make sure each person writes in a distinct style which matches their persona. The situation, conversation, and outcomes can be extremely positive, negative, or very unique. Depictions of conflict are encouraged but are not required. A random news article is given which is very likely unrelated to the company or personas. Use this to free associate and please ignore it in your response if its unrelated."},
                            {"role": "user", "content": f'Simulate a long and fully detailed slack conversation which takes place over 4 days between 2 employees at a company using the following json schema: {str(my_slack_format)} \n\n The list of personas is given: {personas_str} \n\n These people work at the following company: {str(company_info)} \n\n a random news article is given for free association: {title_context}'}
                            ], 
                                extra_body={
                                    "min_p": min_p,
                                    "min_tokens" : min_tokens,
                                    "guided_json": my_slack_format,
                                    "repetition_penalty" : repetition_penalty
                            },
                                temperature = temperature,
                                max_tokens = max_tokens,
                                presence_penalty = presence_penalty,
                                frequency_penalty = frequency_penalty, 
                                n = n
                            )
                        #stop = "<|eot_id|>"
                    except Exception as error:
                        print("ERROR: " + str(error)) ##The dynamics of the errors you see are highly impacted by which constrained generation backend chosen within vllm
                        continue
                    for response in slack_response.choices:
                        #print(response.message.content)
                        #print(type(response.message.content))
                        try:
                            result = json.loads(response.message.content, strict = False)
                            #print(type(result))
                            writer.writerow([str(result["short_situation_summary"]), str(result["slack_channel_name"]), str(result["day1_full_slack_conversation_transcript"]), str(result["day2_full_slack_conversation_transcript"]), str(result["day3_full_slack_conversation_transcript"]), str(result["day4_full_slack_conversation_transcript"]), str(""), str(""), str(""), str(""), str(result["selected_employee_names"]), first_names, str(random_company)])
                        except Exception as error:
                            print("PARSING ERROR: " + str(error))
                            continue



def create_8day_slack_data(max_articles = 200, number_of_conversations = 99999999, persona_file_name = "persona.jsonl", company_file_name = "company.jsonl", input_slack_file_name = "slack_conversations_two_employee.csv", output_slack_file_name = "slack_conversations_eight_day.csv", temperature=1.2, min_p=0.05, min_tokens=6000, repetition_penalty=1.0, max_tokens=6000, presence_penalty=0.05, frequency_penalty=0.05, n=1):
    """
    Extends existing 4-day slack data to 8 days using the given parameters.

    :param max_articles: The maximum number of articles to use for generating data.
    :param number_of_conversations: The number of conversations to extend.
    :param persona_file_name: The name of the file to read the persona data from.
    :param company_file_name: The name of the file to read the company data from.
    :param input_slack_file_name: The name of the file to read the existing 4-day slack data from.
    :param output_slack_file_name: The name of the file to write the extended 8-day slack data to.
    :param temperature: The temperature to use for the model.
    :param min_p: The minimum probability for the model.
    :param min_tokens: The minimum number of tokens for the model.
    :param repetition_penalty: The repetition penalty for the model.
    :param max_tokens: The maximum number of tokens for the model.
    :param presence_penalty: The presence penalty for the model.
    :param frequency_penalty: The frequency penalty for the model.
    :param n: The number of completions to generate.
    """
    client = OpenAI(
        base_url=VLLM_SERVER_URL,
        api_key="token-abc123",
        timeout=999999999,
    )

    import ast

    list_of_articles = []

    for article in crawler.crawl(max_articles=MAX_ARTICLES):
        title_context = str(article.title)
        list_of_articles.append(title_context)

    with open(persona_file_name, 'r') as json_file, open(company_file_name, 'r') as company_json_file:
        json_list = list(json_file)
        company_list = list(company_json_file)

    list_of_data = []
    list_of_company_data = []
    for json_str in json_list:
        try:
            result = json.loads(json_str)
            res = json.loads(result)
            list_of_data.append(res)
        except:
            pass
            
    for json_str in company_list:
        try:
            result = json.loads(json_str)
            res = json.loads(result)
            list_of_company_data.append(res)
        except:
            pass

    with open(input_slack_file_name, 'r', newline='') as input_slack_file, open(output_slack_file_name, 'w', newline='') as output_slack_file:
        reader = csv.reader(input_slack_file, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        writer = csv.writer(output_slack_file, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        header = next(reader)
        writer.writerow(header)
        
        for row in reader:
            goal, channel_name, day1, day2, day3, day4, _, _, _, _, personas, company_workers, company_name = row
            
            class Employee(BaseModel):
                Employee: Literal["Lena Kim", "Bruno Weiss", "Kai Wagner", "Kaitlyn Schmidt", "SofÃ­a RodrÃ­guez", "Hannelore Klose", "Helena Schwarz", "Lysander Machado", "Rohan Kumar", "Kathryn Nguyen", "Colette Dupont", "Andrea Pierre", "Samantha Mokgabudi", "Jamal Ahmed", "Astrid Wagner", "AurÃ©lien Gautier", "Aurora Wong", "Rohini Desai", "Hannelore Kaufmann", "Leopold Morgenstern", "August Mendez", "Ava Kim", "Lukas MÃ¼ller", "Kai Schmidt", "Zara Patel", "Kato Schwarz", "Aparajita Choudhury", "Kai Mei"]
            
            class Slack(BaseModel):
                day5_full_slack_conversation_transcript: str
                day6_full_slack_conversation_transcript: str
                day7_full_slack_conversation_transcript: str
                day8_full_slack_conversation_transcript: str

            my_slack_format = Slack.schema()
            
            company_info = next((d for d in list_of_company_data if d["name"] == company_name), "")
            
            title_context = random.choice(list_of_articles)
            
            prompt = f"""
            Previous 4 days of conversation:
            Day 1: {day1}
            Day 2: {day2}
            Day 3: {day3}
            Day 4: {day4}
            
            Continue this conversation for the next 4 days (days 5-8) between the same employees.
            """
            
            try:
                slack_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are continuing a realistic slack conversation transcript over 4 more days between two employees. Information about the company is given, along with details of each person. Each slack response should start with the author's full name followed by a colon, like this: 'Author: Message'. Make sure each person writes in a distinct style which matches their persona. The situation, conversation, and outcomes can be extremely positive, negative, or very unique. Depictions of conflict are encouraged but are not required. A random news article is given which is very likely unrelated to the company or personas. Use this to free associate and please ignore it in your response if it's unrelated."},
                        {"role": "user", "content": f'{prompt}\n\nContinue the slack conversation for 4 more days using the following json schema: {str(my_slack_format)} \n\n The list of personas is given: {personas} \n\n These people work at the following company: {str(company_info)} \n\n a random news article is given for free association: {title_context}'}
                    ],
                    extra_body={
                        "min_p": min_p,
                        "min_tokens": min_tokens,
                        "guided_json": my_slack_format,
                        "repetition_penalty": repetition_penalty
                    },
                    temperature=temperature,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    n=n
                )
            except Exception as error:
                print("ERROR: " + str(error))
                continue

            for response in slack_response.choices:
                try:
                    result = json.loads(response.message.content, strict=False)
                    writer.writerow([
                        goal, channel_name, day1, day2, day3, day4,
                        str(result["day5_full_slack_conversation_transcript"]),
                        str(result["day6_full_slack_conversation_transcript"]),
                        str(result["day7_full_slack_conversation_transcript"]),
                        str(result["day8_full_slack_conversation_transcript"]),
                        personas, company_workers, company_name
                    ])
                except Exception as error:
                    print("PARSING ERROR: " + str(error))
                    continue



if __name__ == '__main__':
    run(create_company_and_persona_data, create_slack_data, create_8day_slack_data)