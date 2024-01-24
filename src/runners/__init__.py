REGISTRY = {}
from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner
REGISTRY["episode"] = EpisodeRunner
REGISTRY["parallel"] = ParallelRunner

MASKER_REGISTRY = {}
from .masker_episode_runner import EpisodeRunner
from .masker_parallel_runner import ParallelRunner
MASKER_REGISTRY["episode"] = EpisodeRunner
MASKER_REGISTRY["parallel"] = ParallelRunner
