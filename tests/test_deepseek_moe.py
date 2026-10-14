"""CPU dispatch gates; math is covered by the opt-in GPU suite."""
import unittest
from types import SimpleNamespace
try:
    import torch
    import vllm
except ImportError as exc:
    raise unittest.SkipTest("Run in the DeepSeek image") from exc
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from r9700_vllm.compat.deepseek_moe import eligible
from r9700_vllm.moe.deepseek_w4a4 import launch_config


class DeepseekMoEContract(unittest.TestCase):
    def test_geometry_and_isolation(self):
        def tensor(shape,dtype): return torch.empty(shape,dtype=dtype,device='meta')
        expert=SimpleNamespace(_r9700_ds_moe=True,ocp_mx_scheme='w_mxfp4_a_mxfp4',
            activation_config=SimpleNamespace(clamp_limit=10.),
            w1_scale_val=tensor((384,576,160),torch.uint8),
            w2_scale_val=tensor((384,5120,9),torch.uint8))
        args=[expert,tensor((2,5120),torch.bfloat16),
              tensor((384,576,2560),torch.uint8),tensor((384,5120,144),torch.uint8),
              tensor((2,6),torch.float32),tensor((2,6),torch.int32),
              MoEActivation.SILU,384,None,None,None,False]
        self.assertTrue(eligible(*args))
        expert._r9700_ds_moe=False
        self.assertFalse(eligible(*args))
        expert._r9700_ds_moe=True
        for rows in (17,32,128,512,1024,2048):
            args[1]=tensor((rows,5120),torch.bfloat16)
            args[4]=tensor((rows,6),torch.float32)
            args[5]=tensor((rows,6),torch.int32)
            self.assertTrue(eligible(*args))
        args[1]=tensor((2049,5120),torch.bfloat16)
        args[4]=tensor((2049,6),torch.float32)
        args[5]=tensor((2049,6),torch.int32)
        self.assertFalse(eligible(*args))
        args[4]=tensor((2,6),torch.float32)
        args[5]=tensor((2,6),torch.int32)
        args[1]=tensor((2,5120),torch.bfloat16)
        expert.activation_config.clamp_limit=None
        self.assertFalse(eligible(*args))

    def test_tp8_launch_divisibility(self):
        self.assertEqual(launch_config(5120,288),(4,1))
        self.assertEqual(launch_config(576,5120),(2,4))
        with self.assertRaises(ValueError):launch_config(5120,289)

    def test_ep8_geometry(self):
        t=lambda shape,dtype:torch.empty(shape,dtype=dtype,device='meta')
        expert=SimpleNamespace(_r9700_ds_moe=True,ocp_mx_scheme='w_mxfp4_a_mxfp4',
            activation_config=SimpleNamespace(clamp_limit=10.),
            w1_scale_val=t((48,4608,160),torch.uint8),
            w2_scale_val=t((48,5120,72),torch.uint8))
        args=[expert,t((4,5120),torch.bfloat16),t((48,4608,2560),torch.uint8),
              t((48,5120,1152),torch.uint8),t((4,6),torch.float32),
              t((4,6),torch.int32),MoEActivation.SILU,384,t((384,),torch.int32),
              None,None,False]
        self.assertTrue(eligible(*args))
        args[8]=None
        self.assertFalse(eligible(*args))

    def test_dspark_requires_optin_and_separate_geometry(self):
        t=lambda shape,dtype:torch.empty(shape,dtype=dtype,device='meta')
        expert=SimpleNamespace(_r9700_ds_moe=True,_r9700_ds_dspark=False,
            ocp_mx_scheme='w_mxfp4_a_mxfp4',activation_config=SimpleNamespace(clamp_limit=10.),
            w1_scale_val=t((16,4608,160),torch.uint8),w2_scale_val=t((16,5120,72),torch.uint8))
        args=[expert,t((24,5120),torch.bfloat16),t((16,4608,2560),torch.uint8),
              t((16,5120,1152),torch.uint8),t((24,3),torch.float32),t((24,3),torch.int32),
              MoEActivation.SILU,128,t((128,),torch.int32),None,None,False]
        self.assertFalse(eligible(*args))
        expert._r9700_ds_dspark=True
        self.assertTrue(eligible(*args))
        args[7]=384
        self.assertFalse(eligible(*args))
