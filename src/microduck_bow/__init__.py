from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.tasks import MicroduckOnPolicyRunner

from .bow_env_cfg import MicroduckBowRlCfg, make_microduck_bow_env_cfg

register_mjlab_task(
    task_id="Mjlab-Bow-Flat-MicroDuck",
    env_cfg=make_microduck_bow_env_cfg(),
    play_env_cfg=make_microduck_bow_env_cfg(play=True),
    rl_cfg=MicroduckBowRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
