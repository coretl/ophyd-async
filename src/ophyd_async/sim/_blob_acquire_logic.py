from collections.abc import AsyncGenerator

from ophyd_async.core import AsyncStatus, DetectorAcquireLogic, TriggerInfo

from ._pattern_generator import PatternGenerator


class BlobAcquireLogic(DetectorAcquireLogic):
    def __init__(self, pattern_generator: PatternGenerator):
        self.pattern_generator = pattern_generator
        self.trigger_info: TriggerInfo | None = None

    async def acquire(self) -> AsyncGenerator[None, None]:
        # Write the images to file in the background, until done or closed
        async with AsyncStatus(self.pattern_generator.write_images_to_file()) as status:
            yield
            await status

    async def ensure_stopped(self):
        # acquire() owns the writing task, so there is nothing else to stop
        pass
