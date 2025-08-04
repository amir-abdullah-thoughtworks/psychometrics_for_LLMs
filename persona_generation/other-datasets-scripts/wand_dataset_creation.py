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


##Hardcode/Specify everything you can - Ravid

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



crawler = Crawler(*PublisherCollection)
#prompt = f'Write a long and highly detailed user persona using the following json schema: {AnswerFormat.schema_json()} :\n'
my_format = AnswerFormat.schema()
my_company_format = Company.schema()

vllm_url = input("Enter the address of the running vllm server (defaults to http://20.81.188.27:8000/v1)")
if not vllm_url:
    vllm_url = "http://20.81.188.27:8000/v1"

client = OpenAI(
    base_url=vllm_url,
    api_key="token-abc123",
)

# crawl 2 articles and print
company_lines_added = 0
persona_lines_added = 0

company_name = input("Enter the name of the company dataset file (defaults to company.jsonl)")
persona_name = input("Enter the name of the persona dataset file (defaults to persona.jsonl)")

if not company_name:
    company_name = "company.jsonl"

if not persona_name:
    persona_name = "persona.jsonl"

with open(company_name, 'a') as f, open(persona_name, 'a') as g:
    for article in crawler.crawl(max_articles=1000000):
        title_context = str(article.title)
        completion = client.chat.completions.create(model="hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",messages=[
                  {"role": "system", "content": "You invent hypothetical companies. You will be given the title of a news article to use as some creative context for free association of ideas for inventing this hypothetical company. Feel free to invent a company which is entierly different or unrelated to the article title as you please"},
                {"role": "user", "content": f'Write a long and highly detailed description of a hypothetical company using the following json schema: {str(my_company_format)} \n\n The news article title for assisting with this is given here: {title_context}'}
              ], 
                extra_body={
                "guided_json": my_company_format,
                "min_p": 0.5
              },
                temperature = 5.0,
                max_tokens = 500,
                n = 1)
        company = completion.choices[0].message.content
        try:
            company_json = json.loads(company)
        except:
            continue
        print(company)
        f.write(json.dumps(company) + '\n')
        company_lines_added += 1
        if company_lines_added % 10 == 0:
            print(f"Added company line {company_lines_added}")
        job_titles_list = company_json["employee_job_titles"] 
        company_name = company_json["name"] 
        products_list = company_json["products"]
        for another_article in crawler.crawl(max_articles=len(job_titles_list)):
            job_title_to_use = job_titles_list.pop(random.randrange(len(job_titles_list)))
            #product_to_use = random.choice(products_list)
            my_format["properties"]["company_worked_at_name"]["const"] = company_name
            my_format["properties"]["company_worked_at_name"]["enum"] = [company_name]
            my_format["$defs"]["Product"]["properties"]["product"]["enum"] = products_list
            my_format["properties"]["occupation"]["const"] = job_title_to_use
            my_format["properties"]["occupation"]["enum"] = [job_title_to_use]
            title_context = str(another_article.title)
            person_completion = client.chat.completions.create(model="hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",messages=[
                {"role": "system", "content": "You make user personas. You vary everything about the generated persona. Be creative You will be given the title of a news article to use as some creative context for free association of ideas for inventing this hypothetical persona. Feel free to invent a persona which is entierly different or unrelated to the article title as you please"}, ##Condition on *seperate* article text
                {"role": "user", "content": f'Write a long and highly detailed user persona using the following json schema: {str(my_format)} \n\n The news article title for assisting with this is given here: {title_context}'} ## Sample prompts from LIST of prompts (also for companies) - study how this has been done previously
              ], 
                extra_body={
                "guided_json": my_format,
                "min_p": 0.5
              },
                temperature = 5.0,
                max_tokens = 500,
                n = 1
            )
            generated_persona = person_completion.choices[0].message.content
            print(generated_persona)
            try:
                persona_json = json.loads(generated_persona)
            except:
                continue
            g.write(json.dumps(generated_persona) + '\n')
            persona_lines_added += 1
            if persona_lines_added % 10 == 0:
                print(f"Added persona line {persona_lines_added}")

# Create a character level parser and build a transformers prefix function from it
#parser = JsonSchemaParser(AnswerFormat.schema())
#prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

##Option 1
##TODO: Randomly choose (2) eventually 5-10 people, working on same product, to fit in context. Ask model to simulate slack conversation/dialogue history between them over X amount of time. 
##TODO (cont): Append this chat to persona json

##Option 2, Channel
##: For a few product, create a few channels (all in one step at first than do the rest of this), and also create a few general channels
##: Randomly select (2) eventually 3+ people working on same product (or maybe accross? Think of other ways to condition). Ask model to sample channel name, then sample a year of conversations, allowing model to decide who writes/speaks in it. Model has to be reminded to personalize each message sent based on each persona accurately
##: maybe condition based on past conversations generation from Option 1? (Requires big context)

## Think about simulating company at a particular X time, then simulate conversation between Y personas and on their channel over a specific Z time. As time goes on, condition on previous conversations
## Limit size of conversations over X amount of time. Between A and B number of messages multiplied by token limit 
## Think about simulating state of the product directly over time if it's not being done by the LLM naturally

## Think about adding audio/video in the future at the end - Rotem

## Generate questions that a particular persona is asking about a product, or towards other person, or in general (maybe it emerges in the channel simulation?)

