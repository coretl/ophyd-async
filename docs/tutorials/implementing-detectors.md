# Implementing File Writing Detectors

In [](./implementing-devices.md) we learned how to create Devices that talk to a control system to implement a particular behavior in bluesky plans. This behavior was based around the verbs from the [](#bluesky.protocols.Movable) and [](#bluesky.protocols.Readable) protocols, allowing us to use these Devices in a typical scan: moving them to a position, then acquiring data via the control system. We will now explore the [](#bluesky.protocols.WritesExternalAssets) protocol, and how it would be implemented for a File Writing Detector.

## Run the demo

We will return to our [](#ophyd_async.sim) devices we saw in [](./using-devices.md) for this tutorial, and dig a little deeper into what [Event Model Documents](inv:event-model#data_model) they produce. Let's run up our ipython shell again:

```
$ ipython --matplotlib=qt6 -i -m ophyd_async.sim
Python 3.11.11 (main, Dec  4 2024, 20:38:25) [GCC 12.2.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.30.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: 
```

## Run a grid scan and investigate the documents

Now let's run a grid scan on the point detector, and pass a callback to the RunEngine so it prints the documents that are emitted:
```{eval-rst}
.. ipython:: python
    :suppress:

    from ophyd_async.sim.__main__ import *
    # Make the moves faster so docs build don't take too long
    RE(bps.mv(stage.x.velocity, 1000, stage.y.velocity, 1000))

.. ipython:: python
  
    RE(bp.grid_scan([pdet], stage.x, 1, 2, 2, stage.y, 2, 3, 2), print)
```
We see a series of documents being emitted:
- A [](#event_model.RunStart) document that tells us a scan is starting and what sort of scan it is, along with the names of the motors that will be moved.
- An [](#event_model.EventDescriptor) document that tells us that the motor readbacks and detector channels will be all be read together in a single stream. It is used to make the column headings, but it contains more metadata about the Devices too, like their configuration.
- For each point in the scan:
  - An [](#event_model.Event) document, containing the motor readbacks and detector channels with their timestamps. It is used to make each row of the table.
- A [](#event_model.RunStop) document that tells us the scan has stopped, and gives us its status.

Now let's try the same thing, but this time with the blob detector:

```{eval-rst}
.. ipython:: python
    :okwarning:
    
    RE(bp.grid_scan([bdet], stage.x, 1, 2, 2, stage.y, 2, 3, 2), print)
```
This time we see some different documents:
- The same [](#event_model.RunStart) document
- A similar [](#event_model.EventDescriptor) document, but with `'external': 'STREAM:'` on the detector column headings.
- A couple of [](#event_model.StreamResource) documents for each of those detector column headings giving an HDF file name and dataset within it where data will be written.
- For each point in the scan:
  - A couple of [](#event_model.StreamDatum) documents with a range of indices that have been written to an HDF dataset referenced in the StreamResource document.
  - An [](#event_model.Event) document, containing the motor readbacks and timestamps. It is used to make each row of the table.
- The same [](#event_model.RunStop) document

And we can run the plan with both detectors to see a document stream that combines both the previous example:

```{eval-rst}
.. ipython:: python
    :okwarning:
    
    RE(bp.grid_scan([bdet, pdet], stage.x, 1, 2, 2, stage.y, 2, 3, 2), print)
```

## Simplify the plan to just use the detector

The above examples show what happens if you `trigger()` a detector at each point of a scan, in this case taking a single frame each time. Let's write our own simple plan that only triggers and reads from the detector, using the utility [`bps` (`bluesky.plan_stubs`)](inv:bluesky#stub_plans) and [`bpp` (`bluesky.preprocessors`)](inv:bluesky#preprocessors):

```{eval-rst}
.. ipython:: python

    @bpp.stage_decorator([bdet])
    @bpp.run_decorator()
    def my_count_plan():
        for i in range(2):
            yield from bps.trigger_and_read([bdet])

.. ipython:: python
    :okwarning:

    RE(my_count_plan(), print)
```

Here we see the same sort of documents as above, but with only detector information in it.

Note that on each trigger, only a single image is taken, at the default exposure of `0.1s`. If we would like a different exposure time, we can specify with a [](#TriggerInfo):

```{eval-rst}
.. ipython:: python

    from ophyd_async.core import TriggerInfo
  
    @bpp.stage_decorator([bdet])
    @bpp.run_decorator()
    def my_count_plan_with_prepare():
        yield from bps.prepare(bdet, TriggerInfo(livetime=0.001), wait=True)
        for i in range(2):
            yield from bps.trigger_and_read([bdet])

.. ipython:: python
    :okwarning:

    RE(my_count_plan_with_prepare(), print)
```

The [](#TriggerInfo) contains all the parameters needed to set up the detector:
- `trigger` - The trigger type: `INTERNAL` (detector generates its own triggers), `EXTERNAL_EDGE` (rising edge starts an internally-timed exposure), or `EXTERNAL_LEVEL` (high level duration determines exposure time)
- `livetime` - The exposure time per frame
- `deadtime` - The minimum time between exposures (for internal triggering)
- `exposures_per_collection` - Number of exposures averaged into a single collection that is passed to the data writer (default 1)
- `collections_per_event` - Number of collections per bluesky event (default 1)

And if being used for flyscanning (kickoff/complete) then specify:
- `number_of_events` - Number of bluesky events to emit (default 1)

This also moves the work of setting up the detector from the first call of `trigger()` to the `prepare()` call. We can also move the creation of the descriptor earlier, so there is no extra work to do on the first call to `trigger()`:

```{eval-rst}
.. ipython:: python

    from ophyd_async.core import TriggerInfo
  
    @bpp.stage_decorator([bdet])
    @bpp.run_decorator()
    def my_count_plan_with_prepare():
        yield from bps.prepare(bdet, TriggerInfo(), wait=True)
        yield from bps.declare_stream(bdet, name="primary")
        for i in range(2):
            yield from bps.trigger_and_read([bdet])

.. ipython:: python
    :okwarning:

    RE(my_count_plan_with_prepare(), print)
```

## Run a fly scan and investigate the documents

The above demonstrates the detector portion of a step scan, letting the things you want to scan settle before taking data from the detector, and doing this at every point of the scan. Our filewriting detector also supports the ability to fly scan it, taking data while you are scanning other things. To do this, it implements the [](#bluesky.protocols.Flyable) protocol, which allows us to `kickoff()` a series of images, then wait until it is `complete()`:

```{eval-rst}
.. ipython:: python
  
    @bpp.stage_decorator([bdet])
    @bpp.run_decorator()
    def fly_plan():
        yield from bps.prepare(bdet, TriggerInfo(number_of_events=7), wait=True)
        yield from bps.declare_stream(bdet, name="primary")
        yield from bps.kickoff(bdet, wait=True)
        yield from bps.collect_while_completing(flyers=[bdet], dets=[bdet], flush_period=0.5)

.. ipython:: python
    :okwarning:

    RE(fly_plan(), print)
```

As before, we see the start, descriptor, and pair of stream resources, but this time we don't see any event documents. Also, even though we asked for 7 frames from each of the 2 streams, we only got 2 stream datums for each stream. 

What is happening is that instead of triggering, waiting, and publishing a single frame, we are setting up the detector to take 7 frames without stopping, then at the `flush_period` of 0.5s emitting a stream datum with the frames that have been captured. If we inspect the stream datum documents for each stream we see that:
- The first has `'indices': {'start': 0, 'stop': 4}`
- The second has `'indices': {'start': 4, 'stop': 7}`

This behavior allows us to scale up the framerate of the detector without scaling up the number of documents emitted: whether the detector goes at 10Hz or 10MHz it still only emits one stream datum per stream per flush period, just with different numbers in the `indices` field.

## Look at the Device implementations

Now we'll have a look at the code to see how we implement one of these detectors:

### `SimBlobDetector`

```{literalinclude} ../../src/ophyd_async/sim/_blob_detector.py
:language: python
```

It derives from [](#StandardDetector) which is a utility baseclass that implements the protocols we have mentioned so far in this tutorial. It uses three types of logic classes to provide behavior for each protocol verb:
- [](#DetectorTriggerLogic) to setup the exposure and trigger mode of the detector
- [](#DetectorAcquireLogic) to start and stop acquisition and wait for it to complete
- [](#DetectorDataLogic) to tell the detector to open a file, describe the datasets it will write, and emit StreamAsset documents as frames are written

In this case, we have three logic classes written just for this simulation, all taking a reference to the pattern generator that provides methods for both detector control and file writing. In other cases the detector control and filewriting may be handled by different sub-devices that talk to different parts of the control system. The job of the top level detector class is to take the arguments that the logic classes need and, in `logic`, create the logic instances and wire them into a [](#DetectorLogic) by role: `DetectorLogic(trigger=..., acquire=..., data=[...])`.

Now let's look at the underlying classes that define the detector behavior:

### `BlobTriggerLogic`

First we have `BlobTriggerLogic`, a [](#DetectorTriggerLogic) subclass:

```{literalinclude} ../../src/ophyd_async/sim/_blob_trigger_logic.py
:language: python
```

It is decorated with `@dataclass`, which generates `__init__` automatically from the class-level field annotations (here `pattern_generator`). Its job is to configure the detector for different trigger modes. In this case we only implement internal triggering:
- `prepare_internal()` takes the number of frames, livetime (exposure), and deadtime, and sets up the pattern generator with those parameters.

If we wanted to support external triggering, we would also implement:
- `prepare_edge()` for external edge triggering (rising edge starts internally-timed exposure)
- `prepare_level()` for external level/gate triggering (high level duration determines exposure)

We could also implement:
- `get_deadtime()` to calculate the minimum time between exposures. It is given the value
  of every signal the detector reports as configuration, so a signal it needs must be
  declared with [](#StandardReadableFormat.CONFIG_SIGNAL) like on any other
  [](#StandardReadable): the trigger logic does not nominate signals itself
- `default_trigger_info()` to return the [](#TriggerInfo) to use when `trigger()` is called without a preceding `prepare()` (governed by [](#OPHYD_ASYNC_PRESERVE_DETECTOR_STATE)).

### `BlobAcquireLogic`

Next we have `BlobAcquireLogic`, a [](#DetectorAcquireLogic) subclass:

```{literalinclude} ../../src/ophyd_async/sim/_blob_acquire_logic.py
:language: python
```

Its job is to control the acquisition process on the detector, starting and stopping the collection of data:
- `ensure_ready()` (inherited default) delegates to `ensure_stopped`, putting the detector into a known state at the start of each scan.
- `acquire()` is an async generator for one acquisition. It starts the acquisition process that has been prepared, in this case a background task that will write our simulation images to file, yields once it is running, then waits for it to be complete. If the scan is cancelled the detector closes it at the `yield`, so the `finally` cancels the task.
- `ensure_stopped()` stops the detector at the end of a scan, called from `unstage()`. Here `acquire()` owns the only thing running, so there is nothing to do.

### `BlobDataLogic`

Then we have `BlobDataLogic`, an [](#UnboundedDataLogic) subclass:

```{literalinclude} ../../src/ophyd_async/sim/_blob_data_logic.py
:language: python
```

Its job is to manage the file writing and data streaming:
- `start_provider()` works out where the file will go, opens it, and returns a [](#StreamableDataProvider) describing the datasets being written
- `get_hinted_fields()` returns the data keys that are interesting to plot
- `ensure_stopped()` tells the detector to close the file

The [](#StreamableDataProvider) returned from `start_provider()` is made for the scan's collections per event and given `collections_written`, a signal that tracks how many frames have been written. It contains:
- `make_datakeys()` - creates DataKey descriptions for each dataset
- `make_stream_docs()` - emits StreamResource and StreamDatum documents as frames are written

#### Which kind of data a logic produces

The detector starts each of its logics for the scan, and the type of provider that comes back says which kind of data it is.

- a [](#StreamableDataProvider) publishes stream assets, like a file writer. The detector exposes `collect_asset_docs`, and `BlobDataLogic` is an example.
- a [](#PageableDataProvider) publishes event pages, like an areaDetector stats time series or a scaler's arrays, so the detector exposes `collect_pages` instead. Its `make_page()` carries on from where its last page got to; a step scan's `read()` takes the last event of the page made up to the end of the point.
- `None` means "not this stage", for instance a plugin that is switched off. The logic decides this without writing anything. This is not an error: a detector may carry more data logics than any one scan uses.

Separately from that, a data logic is either a [](#BoundedDataLogic) or an [](#UnboundedDataLogic), whichever it inherits from (not both). An unbounded logic, like a file writer, keeps counting for as long as it is told to; its collections-written signal is read when each `prepare()` finishes. A bounded logic needs to be told how many collections to take, like a finite buffer: it is started after the unbounded ones and handed the datakeys their providers already make, and its provider must make none of them. It returns `None` if nothing is left to make, and raises for a scan with no fixed number of collections (`num_collections` of 0) if something is. At each point of a step scan after the first it is stopped and started again, so the buffer is cleared. Either kind may return either kind of provider.

A source that produces one value per event, like a plugin scalar, needs no data logic at all: a detector is a [](#StandardReadable), so register the signal with [](#StandardReadable.set_readable_format).

Both kinds of `start_provider()` are given the detector `name`, which a logic uses for its datakeys (adding its own suffix if it has one), and the fields they need from the scan's [](#TriggerInfo): `collections_per_event`, a `collection_period` and a `flush_period`, and the bounded one also `existing_datakeys` and `num_collections`. The `collection_period` is `exposures_per_collection * (livetime + deadtime)`, where `prepare()` reads the livetime and deadtime back from the trigger logic when the plan's [](#TriggerInfo) leaves `livetime` at 0 and [](#OPHYD_ASYNC_PRESERVE_DETECTOR_STATE) is `YES`; otherwise it is 0 for "use what is on the hardware".

No two providers may make the same datakey: `prepare()` raises if they do. A datakey a provider makes replaces a registered signal of that name.

```{note}
Which logics serve is decided at the first `prepare()` after `stage()` and holds until the next `stage()`, so it can change from one stage to the next: a detector whose file writer is switched off can serve its stats time series instead. Change it **between stages**, not inside one: a run's descriptor is emitted at its start, so switching part way through would leave the descriptor and the documents disagreeing.
```

## Conclusion

We have seen how to make a [](#StandardDetector) and how the [](#DetectorTriggerLogic), [](#DetectorAcquireLogic), and [](#DetectorDataLogic) classes allow us to customise its behavior through composition. This separation of concerns makes it easy to:
- Mix and match different trigger modes, arming strategies, and data outputs
- Add multiple data streams (e.g., multiple HDF writers for different ROIs)
- Combine file writing with signal reading (e.g., add stats plugin outputs)
- Test each component independently

```{seealso}
[](../how-to/implement-ad-detector.md) for writing an implementation of `StandardDetector` for an EPICS areaDetector
```
