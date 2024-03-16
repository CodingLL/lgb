import os, json
import argparse
import pandas as pd
import openai
from tqdm import tqdm
from time import perf_counter
import asyncio
import logging, coloredlogs
from ast import literal_eval

logger = logging.getLogger("sopenai")
logger.setLevel(logging.DEBUG)
coloredlogs.install(level=logging.DEBUG, logger=logger)


async def get_gpt_result_async(messages, temperature=0.7, model='gpt-4'):
    begin = perf_counter()
    for try_idx in range(5):
        try:
            response = await openai.ChatCompletion.acreate(
                messages=messages,
                model=model,
                temperature=temperature,
                api_base="http://cooper.k8s.nb-prod.com/v1",
                # api_key="nb-YYU0mZ1yEZDzsRmTgx5qthLeOMkjLuMCbBMquM8R46y9alCAocrxfCYPn4YviNu5BiE"
                api_key="nb-2YvIn-SBDiIMhNVFDa483SC8h81wbP8z-JvN0tPo3pUi5PVdbIQsowtC2JFead6JXrk",
                stream=False,
                response_format={ "type": "json_object"}
            )
            break
        except Exception as e:
            logger.warning(f'OpenAI Error: {try_idx}, wait for 5 seconds')
            logger.warning(f'OpenAI Error: {e}')
            await asyncio.sleep(5)
    end = perf_counter()
    logger.debug(f'finished GPT query in {end - begin:.3f}s')
    return response.choices[0].message.content.strip()

def get_prompt(claim, rationales):
    prompt = 'Below is a claim to be verified:\n'
    prompt += 'Claim: %s\n' % claim
    prompt += "We search many web pages to verify it. Below are the rationales of these web pages.\n"
    for i, rationale in enumerate(rationales):
        prompt += "{}. Domain: {} Rationale: {}\n".format(i+1, rationale["domain"], rationale["rationale"])
    # prompt += "Rationales: \n"
    # prompt += "\n".join(rationales)
    prompt += "\nPlease judge the claim according to the above rationales, provide a rating and an explanation.\n"
    prompt += "For the rating, it must be on a scale of 0 to 10. Claim totally real corresponds to 10, Claim totally fake corresponds to 0. The more a claim tends towards real, the higher the score.\n"
    prompt += 'Please output with the following json format : {{"rating": XXX, "explanation": YYY}}\n'
    return prompt

class GPTAsync:
    def __init__(self, 
                 total_data, 
                 savepath, 
                 check_model='gpt-4-0125-preview'
    ):
        self.check_model = check_model
        self.prompt = 'Context: {context}\nSentenct: {sentence}\nIs the sentence supported by the context above?\nAnswer Yes or No:'
        self.total_data = total_data
        self.savepath = savepath

    
    async def run_pred_async(self, parallel):
        tasks = []
        sem = asyncio.Semaphore(parallel)

        for i, sample in enumerate(tqdm(self.total_data)):
            tasks.append(asyncio.create_task(self.run_with_semaphore(sem, i, sample)))

            if len(tasks) == args.parallel:
                await asyncio.gather(*tasks)
                tasks = []

        await asyncio.gather(*tasks)

    async def predict(self, i, sample):
        if 'gpt' in sample:
            gpt = sample["gpt"]
        else:
            gpt = sample["nbmodel"]
        gpt = literal_eval(gpt)
        claim = gpt["claim"]
        # print(gpt.keys())
        features = gpt['features']['claim_estimations']
        rationales = [g["rationale"] for g in features]
        
        input_cur = get_prompt(claim, features)
        if rationales:
            messages = [{"role": "user", "content": input_cur}]
            output_cur = await get_gpt_result_async(messages, model = self.check_model)
            output_cur = json.loads(output_cur)
            # if not isinstance(output_cur['rating'], int) or not isinstance(output_cur['rating'], float):
            #     output_cur['rating'] = 5
        else:
            output_cur = {"rating": 5, "explanation": 'no rationale'}
                
        print('\ncnt: {}'.format(i,))
        if i % 1 == 0:
            # logger.info('input: \n', input_cur)
            logger.info('output: {}'.format(output_cur))
            logger.info('label: {}'.format(sample['label']))
        sample['gpt_rating'] = output_cur
        # if i % 1 == 0:
        
        with open(self.savepath, 'a') as f:
            f.write(json.dumps(sample) + '\n')
        return sample
    
    
    async def run_with_semaphore(self, semaphore, i, sample):
        async with semaphore:
            return await self.predict(i, sample)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='data/politifact_547_nbmodel_main_claim_only_v2_0315.csv')
    parser.add_argument("--model", type=str, 
                        choices=["gpt-4-0125-preview","gpt-3.5-turbo-1106"],
                        help="which OpenAI model to choose",
                        default="gpt-4-0125-preview")
    parser.add_argument("--savepath", type=str, default="data/politifact_547_nbmodel_main_claim_only_rationale_v2_0315.jsonl")
    parser.add_argument("--parallel", type=int, default=40)

    args = parser.parse_args()
    done_ids = []
    if os.path.exists(args.savepath):
        with open(args.savepath) as f:
            for line in f:
                dic = json.loads(line)
                done_ids.append(dic['id'])

    print('filepath: {}'.format(args.filepath))
    total_data = []
    df = pd.read_csv(args.filepath)
    for index, row in df.iterrows():
        dic = row.to_dict()
        if dic['id'] in done_ids:
            continue
        total_data.append(dic)
    print('size of total_data: {}'.format(len(total_data)))
    
    GPT = GPTAsync(total_data, 
                   savepath=args.savepath, 
                   check_model=args.model)
    r = asyncio.run(GPT.run_pred_async(parallel=args.parallel))
    
    # save_data = []
    # with open(args.savepath) as f:
    #     for line in f:
    #         dic = json.loads(line)
    #         save_data.append(dic)
    # save_df = pd.DataFrame(save_data)
    # save_df.to_csv(args.savepath[:-6]+'.csv')
    
    # print('csv saved in {}'.format(args.savepath[:-6]+'.csv'))
    
# --------------------------------------------------------
'''
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

        prompt = 'Below is a claim to be verified:\n'
        prompt += 'Claim: %s\n' % claim
        prompt += "We search many web pages to verify it. Below are the rationales of these web pages.\n"
        for i, rationale in enumerate(rationales):
            prompt += "Rationale {}: {}\n".format(i, rationale)
        # prompt += "Rationales: \n"
        # prompt += "\n".join(rationales)
        prompt += "\nPlease judge the claim according to the above rationales, provide a rating and an explanation.\n"
        prompt += "For the rating, it must be on a scale of 0 to 10. Claim totally real corresponds to 10, Claim totally fake corresponds to 0. The more a claim tends towards real, the higher the score.\n"
        prompt += 'Please output with the following json format : {{"rating": XXX, "explanation": YYY}}\n'

        res = gpt_requester_cooper(prompt)
        print(res)
        df.loc[i, "gpt_rating"] = json.dumps(res)

    df.to_csv("politifact_0313_gptrating.csv", index=False)
'''