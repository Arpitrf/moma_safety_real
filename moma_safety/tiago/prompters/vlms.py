"""VLM Helper Functions."""
import base64
import numpy as np
from openai import OpenAI
from anthropic import Anthropic
from termcolor import colored
import moma_safety.utils.utils  as U

import asyncio
import aiohttp
import ssl
import certifi

class Ant:
    def __init__(self, openai_api_key, model_name):
        self.model_name = model_name
        self.client = Anthropic(api_key=openai_api_key)
        assert model_name in ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']

    def create_msg_history(self, history_instruction, history_desc, history_model_analysis, history_imgs):
        messages = []
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': history_instruction}]})
        # TODO: Add instruction response from the assistant.
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'I will pay attention to the history provided below..'}]})

        # Add history of descriptions, imgs, and model analysis. Model analysis is from the assistant.
        assert len(history_desc) == len(history_imgs) == len(history_model_analysis)

        for desc, img, model_analysis in zip(history_desc, history_imgs, history_model_analysis):
            user_content = []
            user_content.append({'type': 'text', 'text': desc})
            if img is not None:
                base64_image_str = base64.b64encode(img).decode('utf-8')
                image_content = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64_image_str}}
                user_content.append(image_content)

            messages.append({'role': 'user', 'content': user_content})
            model_content = []
            model_content.append({'type': 'text', 'text': model_analysis})
            messages.append({'role': 'assistant', 'content': model_content})

        return messages

    def query(self, instruction, prompt_seq, temperature=0, max_tokens=2048, history=None, return_full_chat=False):
        """Queries GPT-4V."""
        messages = []
        # Add instructions as user which are in plain text
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': instruction}]})

        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'sounds good, let me help you with that.'}]})

        if history:
            print("history is provided")
            messages.extend(history)

        # prompt_seq is a list of strings and np.ndarrays
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                content.append({'type': 'text', 'text': elem})
            elif isinstance(elem, np.ndarray):
                base64_image_str = base64.b64encode(elem).decode('utf-8')
                image_content = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64_image_str}}
                content.append(image_content)
        messages.append({'role': 'user', 'content': content})

        # DEBUG:
        # from moma_safety.utils.utils import plot_gpt_chats; plot_gpt_chats([messages], save_key='test', save_dir='temp')
        # import ipdb; ipdb.set_trace()
        error = False
        retry = True
        response = None
        while retry:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    # temperature=temperature,
                    max_tokens=max_tokens
                )
                retry = False
                error = False
            except Exception as e:
                print(f"WARNING!!! ANTHROPIC CHAT FAILED: {str(e)}")
                # import ipdb; ipdb.set_trace()
                error = True
                pass

            if error:
                retry = U.confirm_user(True, 'Press y to retry and n to skip')
            else:
                retry = False

        if error:
            print(colored('Error in querying Anthropic.', 'red'))
            return ""

        if return_full_chat:
            content = response.content[0]
            messages.append({'role': 'assistant', 'content': content})
            return response.content[0].text, messages

        return response.content[0].text

class GPT4V:
    """GPT4V VLM."""

    def __init__(self, openai_api_key, model_name='gpt-4o-2024-05-13'):
        self.model_name = model_name
        self.client = OpenAI(api_key=openai_api_key)

    def create_msg_history(self, history_instruction, history_desc, history_model_analysis, history_imgs):
        messages = []
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': history_instruction}]})
        # TODO: Add instruction response from the assistant.

        # Add history of descriptions, imgs, and model analysis. Model analysis is from the assistant.
        assert len(history_desc) == len(history_imgs) == len(history_model_analysis)

        for desc, img, model_analysis in zip(history_desc, history_imgs, history_model_analysis):
            user_content = []
            user_content.append({'type': 'text', 'text': desc})
            if img is not None:
                base64_image_str = base64.b64encode(img).decode('utf-8')
                image_url = f'data:image/jpeg;base64,{base64_image_str}'
                user_content.append({'type': 'image_url', 'image_url': {'url': image_url}})
            messages.append({'role': 'user', 'content': user_content})

            model_content = []
            model_content.append({'type': 'text', 'text': model_analysis})
            messages.append({'role': 'assistant', 'content': model_content})

        return messages

    async def call_chatgpt_async(self, session, messages, temperature, max_tokens):
        payload = {
            'model': self.model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        try:
            async with session.post(
                url='https://api.openai.com/v1/chat/completions',
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.client.api_key}"},
                json=payload,
                ssl=ssl.create_default_context(cafile=certifi.where())
            ) as response:
                response_json = await response.json()
            if "error" in response_json:
                print(f"OpenAI request failed with error {response_json['error']}")
            return response_json['choices'][0]['message']['content']
        except Exception as e:
            print(f"Request failed with exception: {e}")
            return None

    async def call_chatgpt_bulk(self, messages_list, temperature, max_tokens):
        async with aiohttp.ClientSession() as session:
            # Use asyncio.create_task() to schedule the coroutines
            tasks = [asyncio.create_task(self.call_chatgpt_async(session, messages, temperature, max_tokens)) for messages in messages_list]
            
            # Gather results from all tasks
            responses = await asyncio.gather(*tasks)
        return responses


    def query(self, instruction, prompt_seq, 
              temperature=0, max_tokens=2048, history=None, return_full_chat=False,
              parallel=False, num_parallel=5,
              ):
        """Queries GPT-4V."""
        messages = []
        # Add instructions as user which are in plain text
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': instruction}]})
        # TODO: Add instruction response from the assistant.

        if history:
            print("history is provided")
            messages.extend(history)

        # prompt_seq is a list of strings and np.ndarrays
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                content.append({'type': 'text', 'text': elem})
            elif isinstance(elem, np.ndarray):
                base64_image_str = base64.b64encode(elem).decode('utf-8')
                image_url = f'data:image/jpeg;base64,{base64_image_str}'
                content.append({'type': 'image_url', 'image_url': {'url': image_url}})
        messages.append({'role': 'user', 'content': content})

        for message in messages:
            print(message['role'], message['content'][0]['text'])
            

        # DEBUG:
        # from moma_safety.utils.utils import plot_gpt_chats; plot_gpt_chats([messages], save_key='test', save_dir='temp')
        # import ipdb; ipdb.set_trace()
        error = False
        retry = True
        response = None
        while retry:
            try:
                if parallel: 
                    messages_list = [messages for _ in range(num_parallel)]
                    # Parallel async requests
                    results = asyncio.run(self.call_chatgpt_bulk(messages_list, temperature, max_tokens))
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        # model='gpt-4-vision-preview',
                        # model='gpt-4-turbo-2024-04-09',
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                retry = False
                error = False
            except Exception as e:
                print(f"WARNING!!! OPENAI CHAT FAILED: {str(e)}")
                # import ipdb; ipdb.set_trace()
                error = True
                pass

            if error:
                retry = U.confirm_user(True, 'Press y to retry and n to skip')
            else:
                retry = False

        if error:
            print(colored('Error in querying OpenAI.', 'red'))
            return ""

        if parallel:
            if return_full_chat:
                messages.append({'role': 'assistant', 'content': [response.choices[0].message.content for response in results]})
                return [response for response in results], messages

            return [response for response in results]

        else:
            if return_full_chat:
                messages.append({'role': 'assistant', 'content': response.choices[0].message.content})
                return response.choices[0].message.content, messages

            return response.choices[0].message.content
        

class QWen(GPT4V):
    """QWen VLM."""

    def __init__(self, openai_api_key, model_name='qwen2.5:7b'):
        self.model_name = model_name
        self.client = OpenAI(api_key='ollama',
                             base_url='http://localhost:11434/v1/')
