from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets
    from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups

from opencompass.models.megatron_api import MegatronMoe

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
models=[
    dict(abbr='mamba_moe',
        batch_size=1,
        generation_kwargs=dict(
            temperature=1.0,
            top_k_sampling=0,
            top_p_sampling=0),
        key='xxxxxxxxxxxx',
        max_seq_len=512,
        path='/sharedata/sy/model/map_neo_7b/',
        query_per_second=1,
        type=MegatronMoe),
]
work_dir = './outputs/cmmlu'

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=1,
                task=dict(type=OpenICLInferTask),
                retry=1),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=1,
                task=dict(type=OpenICLEvalTask)),
)

summarizer = dict(
    dataset_abbrs=[
        ['cmmlu', 'accuracy']
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
