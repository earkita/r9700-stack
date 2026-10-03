#!/usr/bin/env python3
"""Bounded GLM image/API checks, not a quality or throughput benchmark.

Requires Pillow. Set LITELLM_MASTER_KEY in the environment. All attempts and
synthetic fixtures are saved to --out; credentials are never saved.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont
from litellm_smoke import anthropic_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    key = os.environ['LITELLM_MASTER_KEY']
    font = ImageFont.truetype('DejaVuSans.ttf', 60)
    fixtures = []
    for i, (code, color) in enumerate(zip(('RAVEN42', 'MAPLE73', 'CEDAR19', 'BIRCH58'),
                                        ('red', 'blue', 'green', 'orange'))):
        # Exercise the full limit (64 x 32 = 2048 tokens) as well as 2016.
        # A square dummy canvas only reaches 2025 and misses this boundary.
        img = Image.new('RGB', (1792, 896) if i == 0 else (1568, 1008), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((40, 50), code, font=font, fill='black')
        draw.rectangle((220, 190, 540, 490), fill=color)
        path = args.out / f'fixture-{i}.png'
        img.save(path)
        fixtures.append((code, base64.b64encode(path.read_bytes()).decode()))

    def image_block(i):
        return dict(type='image', source=dict(type='base64', media_type='image/png', data=fixtures[i][1]))

    question = 'Read the alphanumeric code printed at the top of the image. Return the code.'

    def call(label, path, body, direct=False, expected_status=200):
        base = 'http://127.0.0.1:8080' if direct else 'http://127.0.0.1:4000'
        headers = {'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'}
        if not direct:
            headers['Authorization'] = 'Bearer ' + key
            headers['x-api-key'] = key
        record = dict(request=body, path=path, direct=direct)
        try:
            req = urllib.request.Request(base + path, json.dumps(body).encode(), headers)
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    record['status'] = response.status
                    result = anthropic_stream(response) if body.get('stream') else json.load(response)
            except urllib.error.HTTPError as error:
                record['status'] = error.code
                result = json.loads(error.read())
            record['response'] = result
            assert record['status'] == expected_status, record
            return result
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            (args.out / f'{label}.json').write_text(json.dumps(record, indent=2) + '\n')

    checks = []
    def passed(label):
        checks.append(label)
        (args.out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        print('PASS', label, flush=True)

    common = dict(temperature=1.0, top_p=0.95, max_tokens=2048)
    for direct in (True, False):
        body = dict(model='glm-5.3-flash', **common,
                    chat_template_kwargs=dict(reasoning_effort='high'), messages=[dict(role='user', content=[
                        dict(type='text', text=question),
                        dict(type='image_url', image_url=dict(url='data:image/png;base64,' + fixtures[0][1]))])])
        label = 'direct-openai-image' if direct else 'proxy-openai-image'
        result = call(label, '/v1/chat/completions', body, direct)
        choice = result['choices'][0]
        assert fixtures[0][0] in choice['message']['content'] and choice['finish_reason'] == 'stop', result
        passed(label)

    body = dict(model='glm-5.3-flash-high', **common,
                messages=[dict(role='user', content=[image_block(0), dict(type='text', text=question)])])
    for stream in (False, True):
        result = call(f'anthropic-image-{stream}', '/v1/messages', body | dict(stream=stream))
        answer = ''.join(b.get('text', '') for b in result['content'])
        assert fixtures[0][0] in answer and result['stop_reason'] == 'end_turn', result
        if stream:
            assert result['stream_completed'], result
        passed(f'anthropic-image-{stream}')

    counts = []
    for direct in (True, False):
        count_body = dict(model='glm-5.3-flash' if direct else body['model'], messages=body['messages'])
        counts.append(call(f'count-image-{direct}', '/v1/messages/count_tokens', count_body, direct)['input_tokens'])
    assert counts[0] == counts[1] and counts[0] > 0, counts
    passed('local image token counts match')

    tool = dict(name='Read', description='Read a file', input_schema=dict(type='object',
                properties=dict(file_path=dict(type='string')), required=['file_path']))
    tool_body = body | dict(tools=[tool], stream=True, messages=[
        dict(role='user', content='Read screenshot.png and report the code printed at the top.'),
        dict(role='assistant', content=[dict(type='tool_use', id='toolu_vision_check', name='Read',
                                            input=dict(file_path='screenshot.png'))]),
        dict(role='user', content=[dict(type='tool_result', tool_use_id='toolu_vision_check', content=[image_block(0)])])])
    result = call('anthropic-Read-image-result', '/v1/messages', tool_body)
    assert fixtures[0][0] in ''.join(b.get('text', '') for b in result['content']), result
    assert result['stop_reason'] == 'end_turn' and result['stream_completed'], result
    passed('Claude-style Read image tool result, streaming')

    def concurrent(i):
        request = body | dict(messages=[dict(role='user', content=[image_block(i), dict(type='text', text=question)])])
        result = call(f'concurrent-{i}', '/v1/messages', request)
        text = ''.join(b.get('text', '') for b in result['content'])
        assert fixtures[i][0] in text and result['stop_reason'] == 'end_turn', result
        assert all(code not in text for j, (code, _) in enumerate(fixtures) if j != i), result
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(concurrent, range(4)))
    passed('four concurrent independent image requests')

    multi = body | dict(messages=[dict(role='user', content=[image_block(i) for i in range(4)] +
                        [dict(type='text', text='List the code from each of the four images in order.')])])
    result = call('four-images', '/v1/messages', multi)
    text = ''.join(b.get('text', '') for b in result['content'])
    assert all(code in text for code, _ in fixtures) and result['stop_reason'] == 'end_turn', result
    passed('four images in one request')
    # Count admission at 100 images, with a bounded token budget. This is not
    # a 100 x 2048-token memory/quality qualification.
    small_font = ImageFont.truetype('DejaVuSans.ttf', 28)
    blocks, codes = [], []
    for i in range(100):
        code = f'V{(i * 71 + 17) % 997:03d}Q'
        img = Image.new('RGB', (224, 112), 'white')
        ImageDraw.Draw(img).text((25, 35), code, font=small_font, fill='black')
        path = args.out / f'count-fixture-{i:03d}.png'
        img.save(path)
        codes.append(code)
        blocks.extend([dict(type='text', text=f'Image {i + 1}:'), dict(type='image',
            source=dict(type='base64', media_type='image/png',
                        data=base64.b64encode(path.read_bytes()).decode()))])
    blocks.append(dict(type='text', text='Report the codes in images 1, 50, and 100, in that order.'))
    many = body | dict(messages=[dict(role='user', content=blocks)])
    result = call('100-small-images', '/v1/messages', many)
    text = ''.join(b.get('text', '') for b in result['content'])
    assert result['stop_reason'] == 'end_turn', result
    assert all(codes[i] in text for i in (0, 49, 99)), result
    passed('100 small images admitted; first, middle, last codes retrieved')
    many['messages'][0]['content'].insert(0, image_block(0))
    rejected = call('101-images-rejected', '/v1/messages', many, expected_status=400)
    assert 'At most 100 image(s)' in json.dumps(rejected), rejected
    passed('image count limit enforced')

    text_body = body | dict(messages=[dict(role='user', content='What is 19 + 4? Answer briefly.')])
    result = call('text-after-images', '/v1/messages', text_body)
    assert '23' in ''.join(b.get('text', '') for b in result['content']), result
    assert result['stop_reason'] == 'end_turn', result
    passed('text request after image checks')


if __name__ == '__main__':
    main()
