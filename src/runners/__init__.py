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

AIA_REGISTRY = {}
from .aia_episode_runner import EpisodeRunner
from .aia_parallel_runner import ParallelRunner
AIA_REGISTRY["episode"] = EpisodeRunner
AIA_REGISTRY["parallel"] = ParallelRunner

SAIA_REGISTRY = {}
from .saia_episode_runner import EpisodeRunner
from .saia_parallel_runner import ParallelRunner
SAIA_REGISTRY["episode"] = EpisodeRunner
SAIA_REGISTRY["parallel"] = ParallelRunner
