"""Read-only PCIe evidence; no stress kernel, reset, or clock changes."""
import json
from pathlib import Path
import re
import subprocess


def command(*args):
    return subprocess.check_output(args, text=True, timeout=30)


def hip_mapping(container):
    code = '''import ctypes,json
h=ctypes.CDLL("libamdhip64.so")
n=ctypes.c_int()
assert h.hipGetDeviceCount(ctypes.byref(n))==0
out=[]
for i in range(n.value):
 b=ctypes.create_string_buffer(32)
 assert h.hipDeviceGetPCIBusId(b,32,i)==0
 out.append(dict(hip=i,worker_rank=i,bdf=b.value.decode().lower()))
print(json.dumps(out))'''
    result = json.loads(command('docker', 'exec', container, 'python3', '-c', code))
    if len(result) != 8 or len({x['bdf'] for x in result}) != 8:
        raise RuntimeError('This TP8 protocol requires eight distinct HIP devices')
    return result


def links(mapping):
    devices, paths = {}, {}
    for gpu in mapping:
        endpoint = Path('/sys/bus/pci/devices') / gpu['bdf']
        if not endpoint.exists():
            raise RuntimeError(f"Missing PCI endpoint {endpoint}")
        chain = []
        for node in (endpoint.resolve(), *endpoint.resolve().parents):
            if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', node.name):
                continue
            chain.append(node.name)
            fields = {}
            for key in ('max_link_speed', 'max_link_width', 'current_link_speed',
                        'current_link_width', 'numa_node'):
                raw = (node / key).read_text().strip()
                fields[key] = raw
                if key != 'numa_node' and not re.match(r'^\d+(\.\d+)?', raw):
                    raise RuntimeError(f'Unknown PCIe field {node.name}/{key}: {raw}')
            devices[node.name] = fields
        paths[gpu['bdf']] = chain
    return dict(devices=devices, paths=paths)


def number(value):
    return float(re.match(r'^\d+(\.\d+)?', value).group())


def expected_links(snapshot):
    return {bdf: dict(speed=number(v['max_link_speed']), width=number(v['max_link_width']))
            for bdf, v in snapshot['devices'].items()}


def check_loaded(samples, idle, expected):
    if not samples:
        raise RuntimeError('No PCIe observations during requests')
    for sample in samples:
        if sample['paths'] != idle['paths'] or set(sample['devices']) != set(expected):
            raise RuntimeError('PCIe topology changed')
        for bdf, v in sample['devices'].items():
            original = idle['devices'][bdf]
            if any(v[k] != original[k] for k in ('numa_node', 'max_link_speed', 'max_link_width')):
                raise RuntimeError(f'PCIe capability/NUMA changed: {bdf}')
            if number(v['current_link_width']) != expected[bdf]['width']:
                raise RuntimeError(f'Unexpected negotiated width: {bdf}')
    for bdf, target in expected.items():
        observed = max(number(s['devices'][bdf]['current_link_speed']) for s in samples)
        if observed != target['speed']:
            raise RuntimeError(f'Loaded PCIe speed not established: {bdf}: {observed}, expected {target}')


def capture(out, mapping):
    out.mkdir(parents=True, exist_ok=False)
    for name, args in {
        'devices.json': ('amd-smi', 'list', '--json'),
        'power-limits.json': ('amd-smi', 'static', '--asic', '--limit', '--json'),
        'pcie.json': ('amd-smi', 'metric', '--pcie', '--json'),
        'topology.txt': ('lspci', '-Dtv'),
        'details.txt': ('lspci', '-Dvv'),
    }.items():
        (out / name).write_text(command(*args))
    result = links(mapping)
    (out / 'links.json').write_text(json.dumps(result, indent=2))
    (out / 'mapping.json').write_text(json.dumps(mapping, indent=2))
    return result
