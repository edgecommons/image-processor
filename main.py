"""ImageProcessor entry point -- the EdgeCommons image inference component.

Builds the framework, registers the configuration validator and the component command verbs, then
hands control to the app. The library owns SIGTERM and SIGINT: it flips readiness, unsubscribes,
and closes messaging, while the app own ``stop`` drains the sources, the scheduler, the outbox,
and the ledger in that order.

Run locally (HOST platform, MQTT transport, against a local MQTT broker):

.. code-block:: bash

    python3 main.py --platform HOST --transport MQTT ./test-configs/standalone-messaging.json \
      -c FILE ./test-configs/config.json -t my-thing
"""
import argparse
import logging
import sys

from edgecommons import EdgeCommonsBuilder

from image_processor.commands import DeferredApp, ProcessorCommands
from image_processor.config import register as register_validator
from image_processor.ImageProcessor import ImageProcessor

logger = logging.getLogger("main")


def main():
    arg_parser = argparse.ArgumentParser(
        description="com.mbreissi.edgecommons.ImageProcessor -- an image inference component"
    )
    # add any component specific arguments here

    # The verbs are registered while the runtime builds, so no early request finds a missing
    # verb; the component they act on is bound the moment it exists.
    deferred = DeferredApp()
    commands = ProcessorCommands(deferred)

    builder = (
        EdgeCommonsBuilder.create("com.mbreissi.edgecommons.ImageProcessor")
        .with_args(sys.argv[1:])
        .with_app_options(arg_parser)
        # Nothing this component publishes is anything it consumes: it reads images from a spool
        # or a trigger topic and answers on its own `app` channel. The transport is asked not to
        # echo anyway, so a wildcard trigger filter never picks up this component own output.
        .receive_own_messages(False)
        .configure_commands(commands.register)
    )
    # Reject a configuration that would corrupt evidence or infer on another component files
    # BEFORE it becomes current (validate-then-apply, reject-and-keep).
    register_validator(builder)

    gg = builder.build()
    # Not ready until the executor, the ledger, the sources, the scheduler, the outbox publisher,
    # and at least one activated model generation are all up (the app flips this in run()).
    gg.set_ready(False)

    app = deferred.bind(ImageProcessor(gg))
    try:
        app.run()
    finally:
        app.stop()
        gg.shutdown()


if __name__ == "__main__":
    main()
