"""
Unified LLM client - supporta Anthropic (Claude) e OpenAI (GPT) via Azure AI Foundry.

Usa il campo 'provider' nella configurazione del modello per selezionare il client:
  - 'anthropic': usa AnthropicFoundry (Claude)
  - 'openai':    usa AzureOpenAI (GPT)

Esempio config per Anthropic (default attuale):
    MODEL_PROMPT = {
        'provider': 'anthropic',
        'deployment_name': 'claude-sonnet-4-5',
        'endpoint': 'https://....services.ai.azure.com/anthropic/',
        'api_key': '...',
        'max_tokens': 4096,
        'temperature': 0,
    }

Esempio config per OpenAI/GPT:
    MODEL_PROMPT = {
        'provider': 'openai',
        'deployment_name': 'gpt-4o',
        'endpoint': 'https://....openai.azure.com/',
        'api_key': '...',
        'api_version': '2024-02-15-preview',
        'max_tokens': 4096,
        'temperature': 0,
    }
"""


def call_llm_text(config: dict, system_prompt: str, user_prompt: str) -> tuple:
    """
    Chiama il LLM con input solo testo.

    Returns:
        (response_text: str, usage: dict)
        usage ha sempre le chiavi 'input_tokens' e 'output_tokens'.
    """
    provider = config.get('provider', 'anthropic')

    if provider == 'anthropic':
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(
            api_key=config['api_key'],
            base_url=config['endpoint'],
        )
        msg = client.messages.create(
            model=config['deployment_name'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        usage = {}
        if hasattr(msg, 'usage'):
            usage = {
                'input_tokens': msg.usage.input_tokens,
                'output_tokens': msg.usage.output_tokens,
            }
        return msg.content[0].text, usage

    elif provider == 'openai':
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=config['endpoint'],
            api_key=config['api_key'],
            api_version=config['api_version'],
        )
        # Azure OpenAI GPT-4/5 richiede 'max_completion_tokens' invece di 'max_tokens'
        completion_args = {
            'model': config['deployment_name'],
            'temperature': config['temperature'],
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if 'max_completion_tokens' in config:
            completion_args['max_completion_tokens'] = config['max_completion_tokens']
        elif 'max_tokens' in config:
            completion_args['max_completion_tokens'] = config['max_tokens']
        resp = client.chat.completions.create(**completion_args)
        usage = {}
        if resp.usage:
            usage = {
                'input_tokens': resp.usage.prompt_tokens,
                'output_tokens': resp.usage.completion_tokens,
            }
        return resp.choices[0].message.content, usage

    else:
        raise ValueError(
            f"Provider LLM non supportato: '{provider}'. Valori validi: 'anthropic', 'openai'."
        )


def call_llm_with_image(config: dict, base64_image: str, media_type: str, text_prompt: str) -> str:
    """
    Chiama il LLM con immagine + testo (per analisi immagini/vision).

    Args:
        config:       configurazione modello (MODEL_IMAGE_ANALYSE)
        base64_image: immagine codificata in base64
        media_type:   es. 'image/jpeg', 'image/png'
        text_prompt:  prompt testuale da abbinare all'immagine

    Returns:
        response_text: str
    """
    provider = config.get('provider', 'anthropic')

    if provider == 'anthropic':
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(
            api_key=config['api_key'],
            base_url=config['endpoint'],
        )
        msg = client.messages.create(
            model=config['deployment_name'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )
        return msg.content[0].text

    elif provider == 'openai':
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=config['endpoint'],
            api_key=config['api_key'],
            api_version=config['api_version'],
        )
        resp = client.chat.completions.create(
            model=config['deployment_name'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(
            f"Provider LLM non supportato: '{provider}'. Valori validi: 'anthropic', 'openai'."
        )
