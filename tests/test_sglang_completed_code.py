import importlib.util
from pathlib import Path
import sys

import pytest

bench = Path(__file__).parents[1] / "bench"
sys.path.insert(0, str(bench))
spec = importlib.util.spec_from_file_location("completed_code", bench / "sglang_glm_completed_code.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_reasoning_only_is_not_executable_answer():
    with pytest.raises(ValueError, match="boundary"):
        mod.extract_code("I will implement dependency_waves next")


def test_extract_complete_code_and_reject_extra_prose():
    code = "def dependency_waves(tasks):\n    return []"
    assert mod.extract_code("Thinking</think>```python\n" + code + "\n```") == code
    assert mod.extract_code("Thinking</think>```python\n" + code + "\n```<|user|>") == code
    assert mod.extract_code("Thinking</think>" + code + "<|endoftext|>") == code
    with pytest.raises(ValueError):
        mod.extract_code("Thinking</think>```python\n" + code + "\n```\nDone!")
    with pytest.raises(ValueError):
        mod.extract_code("Thinking</think>```python\n" + code + "\n```<|user|>extra")


def test_harness_checks_wave_boundaries_cycles_and_mutation():
    # Test the independent harness on CPU, without docker/model/GPU work.
    import json
    import subprocess
    implementation = '''
def dependency_waves(tasks):
    pending = {name: set(deps) for name, deps in tasks.items()}
    if any(dep not in pending for deps in pending.values() for dep in deps):
        raise ValueError('unknown')
    done, waves = set(), []
    while pending:
        ready = sorted(name for name, deps in pending.items() if deps <= done)
        if not ready:
            raise ValueError('cycle')
        waves.append(ready)
        done.update(ready)
        for name in ready:
            del pending[name]
    return waves
'''
    good = subprocess.run([sys.executable, "-I", "-S", "-c", mod.HARNESS],
                          input=json.dumps({"code": implementation}),
                          text=True, capture_output=True, timeout=5)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["cases"] >= 160
    bad = subprocess.run([sys.executable, "-I", "-S", "-c", mod.HARNESS],
                         input=json.dumps({"code": "def dependency_waves(tasks): return [sorted(tasks)]"}),
                         text=True, capture_output=True, timeout=5)
    assert bad.returncode != 0
