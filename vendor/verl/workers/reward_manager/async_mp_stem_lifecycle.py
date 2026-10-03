"""Async multiprocessing reward manager with optional STEM verifier sleep."""

from verl import DataProto
from verl.utils.reward_score.stem_llm_judge.lifecycle import StemVerifierLifecycle
from verl.workers.reward_manager import register

from .async_mp import AsyncMultiProcessRewardManager


@register("async_multi_process_stem_lifecycle")
class AsyncMultiProcessStemLifecycleRewardManager(AsyncMultiProcessRewardManager):
    """Wake a local STEM verifier only while LLM-judge rewards are computed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stem_verifier_lifecycle = StemVerifierLifecycle.from_env()

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch:
            return super().__call__(data, return_dict=return_dict)

        data_sources = data.non_tensor_batch.get(self.reward_fn_key, ())
        control_stem_verifier = self.stem_verifier_lifecycle.should_control(data_sources)
        if control_stem_verifier:
            self.stem_verifier_lifecycle.wake_up()

        try:
            return super().__call__(data, return_dict=return_dict)
        finally:
            if control_stem_verifier:
                self.stem_verifier_lifecycle.sleep()
