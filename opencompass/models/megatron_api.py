from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel
import json
from transformers import AutoTokenizer
import torch
import numpy as np

PromptType = Union[PromptList, str]


class MegatronMoe(BaseAPIModel):
    """Model wrapper around 360 GPT.

    Documentations: https://ai.360.com/platform/docs/overview

    Args:
        path (str): Model name
        key (str): Provide API Key
        url (str): Provided URL
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 2.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        path: str,  # model name, e.g.: 360GPT_S2_V9
        key: str,
        url: str = 'http://127.0.0.1:5000/api',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        generation_kwargs: Dict = {
            'temperature': 0.9,
            'max_tokens': 2048,
            'top_p': 0.5,
            'tok_k': 0,
            'repetition_penalty': 1.05,
        }):  # noqa E125
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        self.headers = {
            'Content-Type': 'application/json',
        }
        self.model = path
        self.url = url
        # self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        # if isinstance(input, str):
        #     messages = [{'role': 'user', 'content': input}]
        # else:
        #     messages = []
        #     for item in input:
        #         msg = {'content': item['prompt']}
        #         if item['role'] == 'HUMAN':
        #             msg['role'] = 'user'
        #         elif item['role'] == 'BOT':
        #             msg['role'] = 'assistant'
        #         elif item['role'] == 'SYSTEM':
        #             msg['role'] = 'system'
        #         messages.append(msg)
        # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        data = {"prompts": [input], "tokens_to_generate": max_out_len, "temperature":0.9, "top_p":0.7, "add_BOS":False}

        
        # data.update(self.generation_kwargs)

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            # payload = json.dumps(data)
            try:
                raw_response = requests.request('PUT',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
            except Exception as e:
                self.release()
                print(e)
                max_num_retries += 1
                continue
            response = raw_response.json()
            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue
            if raw_response.status_code == 200:
                msg = response['prompts_plus_generations'][0].strip().replace(input,"")
                print({"prompts": [input], "tokens_to_generate": max_out_len, "response": msg})
                self.logger.debug(f'Generated: {msg}')
                return msg

            # sensitive content, prompt overlength, network error
            # or illegal prompt
            if raw_response.status_code in [400, 401, 402, 429, 500]:
                if 'error' not in response:
                    print(raw_response.status_code)
                    print(raw_response.text)
                    continue
                print(response)
                # tpm(token per minitue) limit
                if response['error']['code'] == '1005':
                    self.logger.debug('tpm limit, ignoring')
                    continue
                elif response['error']['code'] == '1001':
                    msg = '参数错误:messages参数过长或max_tokens参数值过大'
                    self.logger.debug(f'Generated: {msg}')
                    return msg
                else:
                    print(response)

                self.logger.error('Find error message in response: ',
                                  str(response['error']))

            print(raw_response)
            max_num_retries += 1

        raise RuntimeError(raw_response.text)
    
    def get_logits(self, inputs: List[str]):
        data = {"prompts": [prompt for prompt in inputs], "tokens_to_generate": 0, "logprobs": True, "return_logits": True}

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            # payload = json.dumps(data)
            try:
                raw_response = requests.request('PUT',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
            except Exception as e:
                self.release()
                print(e)
                max_num_retries += 1
                continue
            response = raw_response.json()
            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue
            if raw_response.status_code == 200:
                self.logger.debug(f'logits: {response["logits"]}')
                self.logger.debug(f'tokens: {response["tokens"]}')
                return torch.tensor(response["logits"]), torch.tensor(response["tokens"])

            # sensitive content, prompt overlength, network error
            # or illegal prompt
            if raw_response.status_code in [400, 401, 402, 429, 500]:
                if 'error' not in response:
                    print(raw_response.status_code)
                    print(raw_response.text)
                    continue
                print(response)
                # tpm(token per minitue) limit
                if response['error']['code'] == '1005':
                    self.logger.debug('tpm limit, ignoring')
                    continue
                elif response['error']['code'] == '1001':
                    msg = '参数错误:messages参数过长或max_tokens参数值过大'
                    self.logger.debug(f'Generated: {msg}')
                    return msg
                else:
                    print(response)

                self.logger.error('Find error message in response: ',
                                  str(response['error']))

            print(raw_response)
            max_num_retries += 1
            
        raise RuntimeError(raw_response.text)

    def get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        print(inputs[0])
        outputs, inputs = self.get_logits(inputs)
        
        shift_logits = outputs[..., :-1, :].contiguous().float()

        shift_labels = inputs[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss