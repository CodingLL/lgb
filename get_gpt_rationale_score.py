import json

import pandas as pd
import openai
from tqdm import tqdm

df = pd.read_csv("politifact_0313_filter.csv")

def gpt_requester_cooper(prompt, max_tokens=None):
    messages = [
        # {'role': 'system', 'content': 'You are a professional journalist.'},
        {'role': 'user', 'content': prompt}
    ]
    result = None
    for retry in range(5):
        try:
            if max_tokens:
                response = openai.ChatCompletion.create(
                    model='gpt-4-turbo-preview',
                    messages=messages,
                    temperature=0.5,
                    max_tokens=max_tokens,
                    api_base="http://cooper.k8s.nb-prod.com/v1",
                    # api_key="nb-YYU0mZ1yEZDzsRmTgx5qthLeOMkjLuMCbBMquM8R46y9alCAocrxfCYPn4YviNu5BiE"
                    api_key="nb-2YvIn-SBDiIMhNVFDa483SC8h81wbP8z-JvN0tPo3pUi5PVdbIQsowtC2JFead6JXrk"
                )
            else:
                response = openai.ChatCompletion.create(
                    model='gpt-4-turbo-preview',
                    messages=messages,
                    temperature=0.5,
                    api_base="http://cooper.k8s.nb-prod.com/v1",
                    # api_key="nb-YYU0mZ1yEZDzsRmTgx5qthLeOMkjLuMCbBMquM8R46y9alCAocrxfCYPn4YviNu5BiE"
                    api_key="nb-2YvIn-SBDiIMhNVFDa483SC8h81wbP8z-JvN0tPo3pUi5PVdbIQsowtC2JFead6JXrk"
                )
        except Exception as e:
            print("gpt error: %s"%e)
            response = None
        if not response:
            continue
        # print(retry, response)
        try:
            result = response.choices[0].message.content.strip()
            # status = response.choices[0].finish_reason.strip()
        except:
            continue
        if result:
            break
    return result

for i in tqdm(range(len(df))):
    gpt = df.loc[i, "gpt"]
    gpt = json.loads(gpt)
    gpt = gpt["classifier_output"]
    gpt = gpt[0]
    claim = gpt["claim"]
    gpt = gpt["overall_predictions"]
    # print(claim)
    # print(gpt)
    rationales = [g["rationale"] for g in gpt]

    prompt = 'Below is a claim to be verified\n'
    prompt += 'Claim: %s\n' % claim
    prompt += "We search many web pages to verify it. Below are the rationales of these web pages.\n"
    prompt += "Rationales: \n"
    prompt += "\n".join(rationales)
    prompt += "\nPlease judge the claim according to the above rationales, provide a rating and an explanation.\n"
    prompt += "For the rating, it must be on a scale of 0 to 10. Claim totally real corresponds to 10, Claim totally fake corresponds to 0. The more a claim tends towards real, the higher the score.\n"
    prompt += 'Please output with the following json format : {{"rating": XXX, "explanation": YYY}}\n'

    res = gpt_requester_cooper(prompt)
    print(res)
    df.loc[i, "gpt_rating"] = json.dumps(res)

df.to_csv("politifact_0313_gptrating.csv", index=False)


