"""ImageProcessor entry point -- a processing component on edgecommons.

Builds the framework, then hands control to the app: one route per ``component.instances[]`` entry,
each with its own bounded queue, pipeline and worker thread. The library owns SIGTERM/SIGINT ->
graceful shutdown.

Run locally (HOST platform, MQTT transport, against a local MQTT broker):

.. code-block:: bash

    python3 main.py --platform HOST --transport MQTT ./test-configs/standalone-messaging.json \
      -c FILE ./test-configs/config.json -t my-thing
"""
import argparse
import logging
import sys

from edgecommons import EdgeCommonsBuilder

from image_processor.ImageProcessor import ImageProcessor

logger = logging.getLogger("main")


def main():
    arg_parser = argparse.ArgumentParser(description="com.mbreissi.edgecommons.ImageProcessor -- a processing component")
    # add any component specific arguments here

    gg = (
        EdgeCommonsBuilder.create("com.mbreissi.edgecommons.ImageProcessor")
        .with_args(sys.argv[1:])
        .with_app_options(arg_parser)
        # Ask the transport not to hand us our own publishes back. Greengrass IPC honours this;
        # an MQTT broker cannot -- it redelivers our own publishes to our own wildcard subscription
        # like anyone else's. That is why the self-echo guard in image_processor/pipeline.py is not optional.
        .receive_own_messages(False)
        .build()
    )
    # Not ready until the routes are subscribed and running (the app flips this in run()).
    gg.set_ready(False)

    app = ImageProcessor(gg)
    try:
        app.run()
    finally:
        app.stop()
        gg.shutdown()


if __name__ == "__main__":
    main()
