from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ruler.ruler_combined_gen import \
        ruler_combined_datasets
    from opencompass.configs.summarizers.groups.ruler import \
        ruler_summary_groups

from opencompass.models.megatron_api import MegatronMoe

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
print(datasets)
models=[
    dict(abbr='mamba_moe',
        batch_size=1,
        generation_kwargs=dict(
            temperature=1.0,
            top_k_sampling=0,
            top_p_sampling=0),
        key='xxxxxxxxxxxx',
        max_out_len=16384,
        path='/sharedata/sy/model/map_neo_7b/',
        query_per_second=1,
        type=MegatronMoe),
]
work_dir = './outputs/ruler'

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=2),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLInferTask),
                retry=5),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=32,
                task=dict(type=OpenICLEvalTask)),
)

summarizer = dict(
    dataset_abbrs=['ruler_4k', 'ruler_8k', 'ruler_16k', 'ruler_32k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
